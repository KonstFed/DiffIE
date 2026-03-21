"""
Prompt an LLM (via OpenAI API) to extract Open IE triplets from
sentences, then save in CaRB tabbed format for evaluation.

Usage:
    uv run python -m diffopenie.evaluation.llm_extract \
        --input-sentences ../CaRB/data/dev.txt \
        --output-path carb_output/llm_extractions.tsv \
        --gold-examples ../CaRB/data/gold/test.tsv \
        --model gpt-4o
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()

BASE_URL = "https://openrouter.ai/api/v1"
API_KEY = os.environ["API_KEY"]

# fmt: off
# Few-shot examples from CaRB test gold (not dev — avoids contamination).
# Each entry: (sentence, list of (predicate, arg1, arg2, ...))
HARDCODED_EXAMPLES: list[
    tuple[str, list[tuple[str, ...]]]
] = [
    (
        "A Democrat , he became the youngest mayor in"
        " Pittsburgh 's history in September 2006"
        " at the age of 26 .",
        [
            ("became", "he",
             "the youngest mayor in Pittsburgh 's history",
             "in September 2006"),
            ("was", "he", "A Democrat"),
        ],
    ),
    (
        "A cafeteria is also located on the sixth floor ,"
        " a chapel on the 14th floor , and a study hall"
        " on the 15th floor .",
        [
            ("is also located", "A cafeteria",
             "on the sixth floor"),
            ("is located", "a chapel", "on the 14th floor"),
            ("is located", "a study hall",
             "on the 15th floor"),
        ],
    ),
    (
        "A cooling center is a temporary air-conditioned"
        " public space set up by local authorities to"
        " deal with the health effects of a heat wave .",
        [
            ("is a temporary", "A cooling center",
             "air-conditioned public space set up by local"
             " authorities to deal with the health effects"
             " of a heat wave"),
            ("is set up by", "A cooling center",
             "local authorities"),
        ],
    ),
    (
        "According to the 2010 census , the population"
        " of the town is 2,310 .",
        [
            ("According to the 2010 census is",
             "the population of the town", "2,310"),
        ],
    ),
    (
        "On 12 July 2006 , Hezbollah launched a series"
        " of rocket attacks and raids into Israeli"
        " territory , where they killed three Israeli"
        " soldiers and captured a further two .",
        [
            ("launched", "Hezbollah",
             "a series of rocket attacks",
             "into Israeli territory", "On 12 July 2006"),
            ("launched", "Hezbollah",
             "a series of raids",
             "into Israeli territory", "On 12 July 2006"),
            ("killed", "Hezbollah",
             "three Israeli soldiers",
             "in Israeli territory", "On 12 July 2006"),
            ("captured", "Hezbollah",
             "a further two Israeli soldiers",
             "in Israeli territory", "On 12 July 2006"),
        ],
    ),
    (
        "Tom Bradley joined the London , Midland and"
        " Scottish Railway Company as a junior clerk"
        " in the Goods Depot at Kettering in 1941 .",
        [
            ("joined", "Tom Bradley",
             "the London , Midland and Scottish Railway"
             " Company as a junior clerk in the Goods"
             " Depot",
             "at Kettering", "in 1941"),
            ("is", "Tom Bradley",
             "a junior clerk in the Goods Depot",
             "at Kettering", "in 1941"),
        ],
    ),
    (
        "This finding indicated that organic compounds"
        " could carry current .",
        [
            ("indicated that", "This finding",
             "organic compounds could carry current"),
        ],
    ),
    (
        "A spectrum from a single FID has a low"
        " signal-to-noise ratio , but fortunately it"
        " improves readily with averaging of repeated"
        " acquisitions .",
        [
            ("has", "A spectrum from a single FID",
             "a low signal-to-noise ratio"),
            ("improves readily with averaging of",
             "signal-to-noise ratio",
             "repeated acquisitions"),
        ],
    ),
]
# fmt: on


def build_examples_from_gold(
    gold_path: Path, max_sentences: int = 8
) -> list[tuple[str, list[tuple[str, ...]]]]:
    """Parse CaRB gold TSV and group extractions by sentence."""
    from collections import OrderedDict

    sent_exts: OrderedDict[str, list[tuple[str, ...]]] = OrderedDict()
    with open(gold_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            sentence = parts[0]
            extraction = tuple(p.strip() for p in parts[1:])
            sent_exts.setdefault(sentence, []).append(extraction)

    examples = []
    for sent, exts in sent_exts.items():
        if 1 <= len(exts) <= 5:
            examples.append((sent, exts))
        if len(examples) >= max_sentences:
            break
    return examples


def format_examples_for_prompt(
    examples: list[tuple[str, list[tuple[str, ...]]]],
) -> str:
    """Format few-shot examples for the system prompt."""
    parts = []
    for sent, extractions in examples:
        parts.append(f"Sentence: {sent}")
        ext_list = []
        for ext in extractions:
            pred = ext[0]
            args = list(ext[1:])
            ext_list.append(json.dumps({"predicate": pred, "arguments": args}))
        parts.append("Extractions:\n" + "\n".join(ext_list))
        parts.append("")
    return "\n".join(parts)


SYSTEM_PROMPT = """\
You are an Open Information Extraction (Open IE) system. \
Given a sentence, extract ONLY the most important relational \
tuples (predicate, arg1, arg2, ...).

Rules:
- Extract ONLY the most important, salient relations. \
Do NOT extract every possible relation — focus on the \
key facts the sentence conveys.
- Each extraction has a predicate and two or more arguments.
- Arguments should be noun phrases or clauses from the \
sentence.
- The predicate should capture the relation between \
arguments; it can include auxiliary verbs, prepositions, \
or other words needed for meaning.
- Use the exact words from the sentence as much as possible.
- Include temporal and locative modifiers as extra arguments \
when present.
- Output ONLY valid JSON: a list of objects, each with \
"predicate" (string) and "arguments" (list of strings, \
arg1 first).
- If no extractions can be made, output an empty list [].

Here are examples:

{examples}"""


def _extract_single(
    client: OpenAI,
    sentence: str,
    model: str,
    system_msg: str,
    max_retries: int = 3,
) -> tuple[str, list[tuple[str, ...]]]:
    """Extract triplets from a single sentence."""
    user_content = (
        "Extract only the most important Open IE tuples "
        "from the following sentence.\n\n"
        f"Sentence: {sentence}\n\n"
        "Return JSON: "
        '{"extractions": [{"predicate": "...", '
        '"arguments": ["...", ...]}, ...]}'
    )

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            text = response.choices[0].message.content or "{}"
            parsed = json.loads(text)
            extractions = parsed.get("extractions", [])
            tuples = []
            for ext in extractions:
                if isinstance(ext, dict) and "predicate" in ext and "arguments" in ext:
                    pred = ext["predicate"]
                    args = ext["arguments"]
                    if pred and args:
                        tuples.append(tuple([pred] + list(args)))
            return sentence, tuples
        except (json.JSONDecodeError, KeyError) as e:
            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                print(f"  Failed: {sentence[:60]}... ({e})")
                return sentence, []
    return sentence, []


def extract_triplets_llm(
    sentences: list[str],
    model: str,
    examples_text: str,
    workers: int = 8,
    max_retries: int = 3,
    verbose: bool = False,
) -> dict[str, list[tuple[str, ...]]]:
    """Extract triplets using concurrent API calls."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    system_msg = SYSTEM_PROMPT.format(examples=examples_text)
    results: dict[str, list[tuple[str, ...]]] = {}

    def _do(sent: str) -> tuple[str, list[tuple[str, ...]]]:
        client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
        return _extract_single(client, sent, model, system_msg, max_retries)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_do, sent): sent for sent in sentences}
        with tqdm(total=len(sentences), desc="LLM extraction") as pbar:
            for future in as_completed(futures):
                sent, tuples = future.result()
                results[sent] = tuples
                if verbose:
                    print(f"\n  [{sent[:60]}...]")
                    for t in tuples:
                        print(f"    {t}")
                pbar.update(1)

    return results


def write_carb_tsv(
    results: dict[str, list[tuple[str, ...]]],
    output_path: Path,
) -> int:
    """Write results in CaRB tabbed format.

    Format: sentence \\t confidence \\t pred \\t arg1 \\t arg2

    Confidence is ranked by extraction order: 1 - (i-1)/N where
    i is rank (1-indexed) and N is total extractions for that sentence.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for sentence, extractions in results.items():
            n = len(extractions)
            for i, ext in enumerate(extractions):
                pred = ext[0]
                args = list(ext[1:])
                if pred and args:
                    confidence = 1 - i / n if n > 1 else 1.0
                    parts = [sentence, str(confidence), pred]
                    parts.extend(args)
                    f.write("\t".join(parts) + "\n")
                    total += 1
    return total


def main():
    parser = argparse.ArgumentParser(
        description="Extract Open IE triplets using an LLM"
    )
    parser.add_argument(
        "--input-sentences",
        type=Path,
        default=(Path(__file__).resolve().parents[4] / "CaRB" / "data" / "dev.txt"),
        help="Input sentences (one per line)",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("carb_output/llm_extractions.tsv"),
        help="Output TSV file path",
    )
    parser.add_argument(
        "--gold-examples",
        type=Path,
        default=None,
        help="CaRB gold TSV for few-shot examples",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="OpenAI model to use",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of concurrent API calls",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=8,
        help="Number of few-shot examples",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print extractions as they arrive",
    )
    args = parser.parse_args()

    # Build few-shot examples
    if args.gold_examples:
        print(f"Loading examples from {args.gold_examples}...")
        examples = build_examples_from_gold(
            args.gold_examples,
            max_sentences=args.max_examples,
        )
    else:
        print("Using hardcoded few-shot examples...")
        examples = HARDCODED_EXAMPLES[: args.max_examples]
    print(f"  {len(examples)} few-shot example sentences")

    examples_text = format_examples_for_prompt(examples)

    # Read input sentences
    print(f"Reading from {args.input_sentences}...")
    with open(args.input_sentences, encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]
    print(f"  {len(sentences)} sentences to process")

    # Call LLM
    print(f"Extracting with model={args.model}, workers={args.workers}...")
    results = extract_triplets_llm(
        sentences,
        args.model,
        examples_text,
        workers=args.workers,
        verbose=args.verbose,
    )

    # Write output
    total = write_carb_tsv(results, args.output_path)
    print(
        f"\nDone! Wrote {total} extractions from "
        f"{len(sentences)} sentences to {args.output_path}"
    )

    # Print stats
    sentences_with_ext = sum(1 for exts in results.values() if exts)
    avg_ext = total / max(len(sentences), 1)
    print(f"  Sentences with extractions: {sentences_with_ext}/{len(sentences)}")
    print(f"  Average extractions/sentence: {avg_ext:.1f}")

    # Print eval command
    carb_dir = Path(__file__).resolve().parents[4] / "CaRB"
    out_abs = args.output_path.resolve()
    print("\nTo evaluate with CaRB:")
    print(f"  cd {carb_dir}")
    print(
        f"  python carb.py --gold=data/gold/dev.tsv --out=/dev/null --tabbed={out_abs}"
    )


if __name__ == "__main__":
    main()

"""Prompt an LLM (OpenAI-compatible API) to extract Open IE triplets, write them
in CaRB tabbed format, and report extraction speed.

Points at any OpenAI-compatible server via env vars:
    OPENAI_BASE_URL   (default http://localhost:8000/v1 — a local vLLM/SGLang server)
    API_KEY           (default "EMPTY"; local servers ignore it)

Deploy the server on an A100 with scripts/serve_qwen_vllm.slurm.

Usage:
    OPENAI_BASE_URL=http://<node>:8000/v1 API_KEY=EMPTY \
    uv run --with openai python -m diffopenie.evaluation.llm_extract \
        --input-sentences benchmarks/CaRB/data/dev.txt \
        --output-path carb_output/qwen_dev.tsv \
        --gold benchmarks/CaRB/data/gold/dev.tsv \
        --model Qwen/Qwen2.5-7B-Instruct \
        --workers 8 --limit 200 --timing-out carb_output/qwen_timing.json

Fairness notes:
- Report on CaRB dev (few-shot exemplars below are from CaRB *test* gold, so they
  do not contaminate a dev evaluation). If you switch to test, pass
  --gold-examples benchmarks/CaRB/data/gold/dev.tsv instead.
- For a latency number comparable to timing_benchmark (single stream), run with
  --workers 1. Use --workers >1 only to report concurrent server throughput.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1")
API_KEY = os.environ.get("API_KEY", "EMPTY")


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
- Extract ONLY the most important, salient relations. Do NOT extract every \
possible relation — focus on the key facts the sentence conveys.
- Each extraction has a predicate and two or more arguments.
- Arguments should be noun phrases or clauses from the sentence.
- The predicate should capture the relation between arguments; it can include \
auxiliary verbs, prepositions, or other words needed for meaning.
- Use the exact words from the sentence as much as possible.
- Include temporal and locative modifiers as extra arguments when present.
- Output ONLY valid JSON: {"extractions": [{"predicate": "...", \
"arguments": ["...", ...]}, ...]}. If none, output {"extractions": []}.

Here are examples:

{examples}"""


def _extract_single(
    client: OpenAI,
    sentence: str,
    model: str,
    system_msg: str,
    max_retries: int = 3,
) -> tuple[str, list[tuple[str, ...]]]:
    user_content = (
        "Extract only the most important Open IE tuples from the following "
        f"sentence.\n\nSentence: {sentence}\n\n"
        'Return JSON: {"extractions": [{"predicate": "...", '
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
                    pred, args = ext["predicate"], ext["arguments"]
                    if pred and args:
                        tuples.append(tuple([pred] + list(args)))
            return sentence, tuples
        except (json.JSONDecodeError, KeyError, Exception) as e:  # noqa: BLE001
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
) -> dict[str, list[tuple[str, ...]]]:
    system_msg = SYSTEM_PROMPT.format(examples=examples_text)
    results: dict[str, list[tuple[str, ...]]] = {}

    def _do(sent: str):
        client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
        return _extract_single(client, sent, model, system_msg)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_do, s): s for s in sentences}
        with tqdm(total=len(sentences), desc="LLM extraction") as pbar:
            for future in as_completed(futures):
                sent, tuples = future.result()
                results[sent] = tuples
                pbar.update(1)
    return results


def write_carb_tsv(
    results: dict[str, list[tuple[str, ...]]], output_path: Path
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for sentence, extractions in results.items():
            n = len(extractions)
            for i, ext in enumerate(extractions):
                pred, args = ext[0], list(ext[1:])
                if pred and args:
                    confidence = 1 - i / n if n > 1 else 1.0
                    f.write("\t".join([sentence, str(confidence), pred, *args]) + "\n")
                    total += 1
    return total


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract Open IE triplets with an LLM")
    ap.add_argument("--input-sentences", type=Path,
                    default=Path("benchmarks/CaRB/data/dev.txt"))
    ap.add_argument("--output-path", type=Path,
                    default=Path("carb_output/llm_extractions.tsv"))
    ap.add_argument("--gold", type=Path, default=None,
                    help="CaRB gold TSV; if given, print in-house CaRB metrics")
    ap.add_argument("--gold-examples", type=Path, default=None,
                    help="CaRB gold TSV to draw few-shot exemplars from "
                         "(must NOT be the split you evaluate on)")
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--workers", type=int, default=8,
                    help="Concurrent requests. Use 1 for single-stream latency.")
    ap.add_argument("--max-examples", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None,
                    help="Only process the first N sentences (match timing subset).")
    ap.add_argument("--timing-out", type=Path, default=None,
                    help="Write a timing JSON (wall-clock, sec/sent, sent/s).")
    args = ap.parse_args()

    if args.gold_examples:
        examples = build_examples_from_gold(args.gold_examples, args.max_examples)
        print(f"Few-shot: {len(examples)} exemplars from {args.gold_examples}")
    else:
        raise SystemExit(
            "Pass --gold-examples <CaRB gold TSV> from a split you are NOT "
            "evaluating on (e.g. gold/test.tsv when reporting on dev)."
        )
    examples_text = format_examples_for_prompt(examples)

    with open(args.input_sentences, encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]
    if args.limit is not None:
        sentences = sentences[: args.limit]
    print(f"{len(sentences)} sentences | endpoint={BASE_URL} | model={args.model} "
          f"| workers={args.workers}")

    start = time.perf_counter()
    results = extract_triplets_llm(sentences, args.model, examples_text, args.workers)
    elapsed = time.perf_counter() - start

    total = write_carb_tsv(results, args.output_path)
    n_sent = len(sentences)
    sec_per_sent = elapsed / max(n_sent, 1)
    sent_per_sec = n_sent / elapsed if elapsed else 0.0

    print(f"\nWrote {total} extractions from {n_sent} sentences to {args.output_path}")
    print(f"  avg extractions/sentence: {total / max(n_sent, 1):.2f}")
    print("--- speed ---")
    print(f"  workers:            {args.workers}"
          f"{'  (single-stream latency)' if args.workers == 1 else '  (concurrent throughput)'}")
    print(f"  total wall-clock:   {elapsed:.2f} s")
    print(f"  sec/sentence:       {sec_per_sent:.4f}")
    print(f"  sentences/second:   {sent_per_sec:.3f}")

    if args.timing_out:
        args.timing_out.parent.mkdir(parents=True, exist_ok=True)
        args.timing_out.write_text(json.dumps({
            "model": args.model, "workers": args.workers, "num_sentences": n_sent,
            "total_seconds": elapsed, "sec_per_sentence": sec_per_sent,
            "sentences_per_second": sent_per_sec, "endpoint": BASE_URL,
        }, indent=2))
        print(f"  timing JSON -> {args.timing_out}")

    if args.gold:
        from diffopenie.evaluation.carb_metrics import (
            evaluate as carb_evaluate,
            load_gold_file,
            load_predicted_file,
        )
        gold = load_gold_file(str(args.gold))
        predicted = load_predicted_file(str(args.output_path))
        res = carb_evaluate(gold, predicted)
        print("--- CaRB (in-house) ---")
        print(f"  AUC={res.auc:.3f}  P={res.precision:.3f}  "
              f"R={res.recall:.3f}  F1={res.f1:.3f}")


if __name__ == "__main__":
    main()

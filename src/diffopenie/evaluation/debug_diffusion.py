"""Debug tool: visualize diffusion reverse steps on dataset examples.

Shows ground-truth labels vs. model generation at every reverse timestep,
so you can see exactly where the diffusion process succeeds or fails.

Usage:
    uv run python -m diffopenie.evaluation.debug_diffusion \
        --config config/training_mdlm.yaml \
        --checkpoint-path checkpoints/checkpoint_best.pt \
        --dataset lsoie --split validation --index 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from diffopenie.data.lsoie import SequenceLSOEIDataset, labels_to_indices
from diffopenie.training.train_example import TrainingConfig
from diffopenie.utils import load_config

# ANSI colors
_C = {
    "B": "\033[0m",      # reset (background)
    "S": "\033[96m",     # cyan  (subject)
    "R": "\033[95m",     # magenta (relation)
    "O": "\033[92m",     # green (object)
    "M": "\033[90m",     # grey
}
_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"


_DEBUG_ID2STR = {"0": "B", "1": "S", "2": "R", "3": "O", "4": "M"}


def state_id_to_str(sid: int) -> str:
    return _DEBUG_ID2STR.get(str(sid), "?")


def colorize_tag(tag: str) -> str:
    return f"{_C.get(tag, _RESET)}{tag}{_RESET}"


def colorize_token_tag(token: str, tag: str) -> str:
    c = _C.get(tag, _RESET)
    return f"{c}{token}{_RESET}"


def load_model(config: TrainingConfig, checkpoint_path: Path):
    model = config.model.create()
    trainer = config.trainer.create(model=model)
    trainer.load_checkpoint(checkpoint_path)
    return trainer.model


def print_header(title: str) -> None:
    print(f"\n{'=' * 80}")
    print(f"  {_BOLD}{title}{_RESET}")
    print(f"{'=' * 80}")


def print_aligned_row(
    tokens: list[str], tags: list[str], label: str, widths: list[int],
) -> None:
    """Print one row: label + tokens colored by tags, padded to widths."""
    parts = []
    for tok, tag, w in zip(tokens, tags, widths):
        colored = colorize_token_tag(tok, tag)
        # pad based on raw token length (color codes are invisible)
        pad = w - len(tok)
        parts.append(colored + " " * max(pad, 0))
    print(f"  {label:>8s}  {' '.join(parts)}")


def print_tag_row(tags: list[str], label: str, widths: list[int]) -> None:
    """Print one row of tags only."""
    parts = []
    for tag, w in zip(tags, widths):
        colored = colorize_tag(tag)
        pad = w - len(tag)
        parts.append(colored + " " * max(pad, 0))
    print(f"  {label:>8s}  {' '.join(parts)}")


def _span_text(words: list[str], span) -> str:
    """Extract text from a (start, end) word span."""
    if span is None or span[0] is None:
        return "---"
    s, e = span
    return " ".join(words[s : e + 1])


def _print_all_gt_lsoie(
    words: list[str],
    all_labels: list[list[str]],
) -> None:
    """Print all GT triplets from LSOIE label sequences."""
    print(f"\n  {_BOLD}All GT triplets ({len(all_labels)}):{_RESET}")
    for i, labels in enumerate(all_labels):
        sub_span, obj_span, pred_span = labels_to_indices(labels)
        sub = _span_text(words, sub_span)
        obj = _span_text(words, obj_span)
        pred = _span_text(words, pred_span)
        print(
            f"    {i + 1}. "
            f"{colorize_tag('S')} {sub}  "
            f"{colorize_tag('R')} {pred}  "
            f"{colorize_tag('O')} {obj}"
        )


def _print_all_gt_carb(
    triplets: list[tuple[str, str, str]],
) -> None:
    """Print all GT triplets from CaRB gold (sub, pred, obj text)."""
    print(f"\n  {_BOLD}All GT triplets ({len(triplets)}):{_RESET}")
    for i, (sub, pred, obj) in enumerate(triplets):
        print(
            f"    {i + 1}. "
            f"{colorize_tag('S')} {sub}  "
            f"{colorize_tag('R')} {pred}  "
            f"{colorize_tag('O')} {obj}"
        )


def debug_single(
    model,
    tokens: list[str],
    token_ids: list[int],
    gt_labels: torch.Tensor | None,
    words: list[str] | None = None,
    all_gt_lsoie: list[list[str]] | None = None,
    all_gt_carb: list[tuple[str, str, str]] | None = None,
) -> None:
    """Run reverse diffusion on one example, printing every step."""
    device = model.device
    T = model.scheduler.num_steps

    # Encode
    ids_t = torch.tensor([token_ids], dtype=torch.long, device=device)
    attn = torch.ones_like(ids_t)
    token_embeddings = model.encode_tokens(ids_t, attn)

    # Run generate with intermediates
    x_0, intermediates = model.generate(
        batch_size=1,
        token_embeddings=token_embeddings,
        attention_mask=attn,
        return_intermediate=True,
    )
    # intermediates: [1, L, T]
    #   dim2 index 0 = after step t=T, ..., T-1 = x_0
    intermediates = intermediates[0].cpu()  # [L, T]
    x_0 = x_0[0].cpu()  # [L]

    # Print all GT triplets for this sentence
    if all_gt_lsoie is not None and words is not None:
        _print_all_gt_lsoie(words, all_gt_lsoie)
    if all_gt_carb is not None:
        _print_all_gt_carb(all_gt_carb)

    # Column widths: just wide enough for token or tag (max 4 chars)
    col_w = [max(len(tok), 1) for tok in tokens]

    # Print sentence
    if words is not None:
        print(f"\n  {'Sentence':>8s}  {' '.join(words)}")
    print_aligned_row(tokens, ["B"] * len(tokens), "Tokens", col_w)

    # Print ground truth if available
    if gt_labels is not None:
        gt_tags = [state_id_to_str(int(s)) for s in gt_labels]
        print_tag_row(gt_tags, "GT", col_w)

    print(f"\n  {_DIM}--- Reverse diffusion ({T} steps) ---{_RESET}")

    # Print each reverse step
    for step_idx in range(T):
        ti = T - step_idx  # t=T, T-1, ..., 1
        step_states = intermediates[:, step_idx]  # [L]
        tags = [state_id_to_str(int(s)) for s in step_states]
        print_tag_row(tags, f"t={ti:>3d}", col_w)

    # Final result
    final_tags = [state_id_to_str(int(s)) for s in x_0]
    print()
    print_tag_row(final_tags, "Final", col_w)
    if gt_labels is not None:
        gt_tags = [state_id_to_str(int(s)) for s in gt_labels]
        print_tag_row(gt_tags, "GT", col_w)

    # Accuracy
    if gt_labels is not None:
        correct = (x_0 == gt_labels).sum().item()
        total = len(gt_labels)
        print(
            f"\n  Accuracy: {correct}/{total}"
            f" ({100 * correct / total:.1f}%)"
        )

        # Per-class breakdown
        for tag_name, tag_id in [
            ("B", 0), ("S", 1), ("R", 2), ("O", 3),
        ]:
            gt_mask = gt_labels == tag_id
            if gt_mask.sum() == 0:
                continue
            tag_correct = ((x_0 == tag_id) & gt_mask).sum().item()
            tag_total = gt_mask.sum().item()
            print(f"    {tag_name}: {tag_correct}/{tag_total}")

    # Human-readable triplet comparison
    print(f"\n  {_BOLD}Triplet summary:{_RESET}")
    if words is not None:
        _print_triplet_from_tags(words, final_tags, "Pred", tokens)
        if gt_labels is not None:
            _print_triplet_from_tags(words, gt_tags, "GT", tokens)


def _print_triplet_from_tags(
    words: list[str],
    tags: list[str],
    label: str,
    tokens: list[str],
) -> None:
    """Print S/R/O spans in word-level text."""
    # Collect token-level spans per role
    for role, role_name in [("S", "Subject"), ("R", "Relation"), ("O", "Object")]:
        role_tokens = [tokens[i] for i, t in enumerate(tags) if t == role]
        if role_tokens:
            text = " ".join(role_tokens).replace(" ##", "")
            print(f"    {label} {colorize_tag(role)} {role_name}: {text}")
        else:
            print(f"    {label} {colorize_tag(role)} {role_name}: ---")


def load_carb_data(
    sentences_path: Path, gold_path: Path | None,
) -> tuple[list[str], dict[str, list[tuple[str, str, str]]]]:
    """Load CaRB sentences and optional gold triplets.

    Returns:
        (sentences, gold_map) where gold_map maps sentence
        to list of (subject, predicate, object) tuples.
    """
    with open(sentences_path, "r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]

    gold_map: dict[str, list[tuple[str, str, str]]] = {}
    if gold_path is not None and gold_path.exists():
        with open(gold_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) < 4:
                    continue
                sent = parts[0]
                pred, subj, obj = parts[1], parts[2], parts[3]
                gold_map.setdefault(sent, []).append(
                    (subj, pred, obj)
                )
    return sentences, gold_map


def main():
    parser = argparse.ArgumentParser(
        description="Debug diffusion: visualize reverse steps"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        choices=["lsoie", "carb"],
        default="lsoie",
        help="Dataset to load examples from",
    )
    parser.add_argument("--split", default="validation")
    parser.add_argument(
        "--index", type=int, default=0,
        help="Dataset/sentence index to inspect",
    )
    parser.add_argument(
        "--sentence",
        type=str,
        default=None,
        help="Custom sentence instead of dataset",
    )
    parser.add_argument(
        "--carb-dir",
        type=Path,
        default=None,
        help="Path to CaRB repo (contains data/)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    print("Loading model...")
    config = load_config(TrainingConfig, args.config)
    model = load_model(config, args.checkpoint_path)
    model.eval()
    tokenizer = model.encoder.tokenizer

    if args.sentence is not None:
        # Custom sentence — no ground truth
        words = args.sentence.split()
        encoding = tokenizer(
            words,
            is_split_into_words=True,
            add_special_tokens=False,
        )
        toks = tokenizer.convert_ids_to_tokens(
            encoding["input_ids"]
        )
        print_header(f"Custom: {args.sentence}")
        with torch.no_grad():
            debug_single(
                model, toks, encoding["input_ids"],
                gt_labels=None, words=words,
            )

    elif args.dataset == "carb":
        # CaRB dataset
        carb_dir = args.carb_dir
        if carb_dir is None:
            # Default: sibling directory ../CaRB relative to project
            carb_dir = (
                Path(__file__).resolve().parents[3] / "CaRB"
            )
        split = args.split
        if split == "validation":
            split = "dev"
        sent_path = carb_dir / "data" / f"{split}.txt"
        gold_path = carb_dir / "data" / "gold" / f"{split}.tsv"

        if not sent_path.exists():
            raise FileNotFoundError(
                f"CaRB sentences not found: {sent_path}"
            )

        sentences, gold_map = load_carb_data(sent_path, gold_path)

        if args.index >= len(sentences):
            raise IndexError(
                f"Index {args.index} out of range "
                f"(dataset has {len(sentences)} sentences)"
            )

        sentence = sentences[args.index]
        words = sentence.split()
        gt_carb = gold_map.get(sentence, [])

        encoding = tokenizer(
            words,
            is_split_into_words=True,
            add_special_tokens=False,
        )
        toks = tokenizer.convert_ids_to_tokens(
            encoding["input_ids"]
        )

        print_header(f"[CaRB {split}] idx={args.index}: {sentence}")
        with torch.no_grad():
            debug_single(
                model, toks, encoding["input_ids"],
                gt_labels=None, words=words,
                all_gt_carb=gt_carb if gt_carb else None,
            )

    else:
        # LSOIE dataset
        tokenizer_name = config.model.encoder.model_name
        ds = SequenceLSOEIDataset(
            split=args.split, tokenizer_name=tokenizer_name,
        )
        item = ds[args.index]
        toks = item["tokens"]
        token_ids = item["token_ids"]
        gt_labels = item["labels"]

        row = ds.dataset.iloc[args.index]
        words = row["words"]
        sentence = " ".join(words)

        # Find ALL GT label sequences for this sentence
        matches = ds.dataset[
            ds.dataset["sentence"] == sentence
        ]
        all_gt = matches["label"].tolist()

        print_header(
            f"[{args.split}] idx={args.index}: {sentence}"
        )
        with torch.no_grad():
            debug_single(
                model, toks, token_ids, gt_labels,
                words=words, all_gt_lsoie=all_gt,
            )


if __name__ == "__main__":
    main()

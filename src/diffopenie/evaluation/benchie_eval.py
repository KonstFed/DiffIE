"""Evaluate a trained DiffOpenIE model on the BenchIE benchmark.

Writes predictions in BenchIE's tab-separated format
    sent_id <TAB> subject <TAB> relation <TAB> object
where sent_id is the 1-indexed line number of the input sentences file,
then scores against the BenchIE gold annotations.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from diffopenie.evaluation.carb_eval import extract_span_text, load_model
from diffopenie.training.train_example import TrainingConfig
from diffopenie.utils import load_config, seed_everything


def load_benchie_cls(benchie_root: Path):
    """Import BenchIE's `Benchie` class.

    BenchIE uses bare-name imports (`from gold_annotations import ...`), so we
    add its `src/` to sys.path before importing.
    """
    src_dir = benchie_root / "src"
    benchie_path = src_dir / "benchie.py"
    if not benchie_path.exists():
        raise FileNotFoundError(f"benchie.py not found at {benchie_path}")
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from benchie import Benchie  # noqa: PLC0415

    return Benchie


def load_sentences(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


@torch.no_grad()
def predict_benchie(model, sentences: list[str]) -> list[tuple[int, str, str, str]]:
    """Return (sent_id, subject, relation, object) rows; sent_id is 1-indexed."""
    rows: list[tuple[int, str, str, str]] = []
    for idx, sentence in enumerate(tqdm(sentences, desc="Predicting"), start=1):
        words = sentence.split()
        triplets, _probs = model.get_carb_prediction(words)
        for sub_span, obj_span, pred_span in triplets:
            subject = extract_span_text(words, sub_span)
            object_ = extract_span_text(words, obj_span)
            relation = extract_span_text(words, pred_span)
            if subject and relation and object_:
                rows.append((idx, subject, relation, object_))
    return rows


def write_predictions(rows: list[tuple[int, str, str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for sent_id, subj, rel, obj in rows:
            f.write(f"{sent_id}\t{subj}\t{rel}\t{obj}\n")


def report(
    Benchie,
    gold_path: Path,
    predictions_path: Path,
    system_name: str = "diffopenie",
):
    benchie = Benchie()
    benchie.load_gold_annotations(filename=str(gold_path))
    benchie.add_oie_system_extractions(
        oie_system_name=system_name, filename=str(predictions_path)
    )
    benchie.compute_precision()
    benchie.compute_recall()
    benchie.compute_f1()
    scores = benchie.scores[system_name]
    print(f"\nBenchIE results for {system_name}:")
    print(f"  Precision: {scores.precision:.3f}")
    print(f"  Recall:    {scores.recall:.3f}")
    print(f"  F1:        {scores.f1:.3f}")
    return {
        "precision": scores.precision,
        "recall": scores.recall,
        "f1": scores.f1,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained DiffOpenIE model on the BenchIE benchmark"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument(
        "--benchie-root",
        type=Path,
        default=Path("benchmarks/benchie"),
        help="Path to the BenchIE repo (containing src/ and data/)",
    )
    parser.add_argument(
        "--language",
        default="en",
        choices=["en", "de", "zh"],
        help="BenchIE language split to evaluate against",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchie_output"),
        help="Where to write predictions TSV",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help="Optional pre-computed predictions TSV; if set, "
        "skips model inference and just re-scores",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Run inference once per seed and save per-seed metrics JSON",
    )
    args = parser.parse_args()

    Benchie = load_benchie_cls(args.benchie_root)
    config = load_config(TrainingConfig, args.config)

    if args.language == "en":
        gold_path = (
            args.benchie_root
            / "data"
            / "gold"
            / "2_annotators"
            / "benchie_gold_annotations_en.txt"
        )
    else:
        gold_path = (
            args.benchie_root
            / "data"
            / "gold"
            / f"benchie_gold_annotations_{args.language}.txt"
        )
    sentences_path = (
        args.benchie_root / "data" / "sentences" / f"sample300_{args.language}.txt"
    )

    if args.seeds:
        print("Loading model...")
        model = load_model(config, args.checkpoint_path)
        model.eval()
        sentences = load_sentences(sentences_path)
        print(f"Processing {len(sentences)} sentences from {sentences_path}")
        all_metrics: list[dict] = []
        for seed in args.seeds:
            seed_everything(seed)
            rows = predict_benchie(model, sentences)
            predictions_path = args.output_dir / f"diffopenie_{args.language}_explicit_{seed}.txt"
            write_predictions(rows, predictions_path)
            metrics = report(Benchie, gold_path, predictions_path, system_name=f"diffopenie_{seed}")
            with open(args.output_dir / f"metrics_{seed}.json", "w") as f:
                json.dump(metrics, f, indent=2)
            all_metrics.append(metrics)
            print(f"Seed {seed}: P={metrics['precision']:.3f}  R={metrics['recall']:.3f}  F1={metrics['f1']:.3f}")
        keys = ["precision", "recall", "f1"]
        print("\n--- mean ± std ---")
        for k in keys:
            vals = np.array([m[k] for m in all_metrics])
            print(f"  {k:<12} {vals.mean():.3f} ± {vals.std():.3f}")
        return

    if args.predictions is not None:
        print(f"Using predictions from {args.predictions}")
        predictions_path = args.predictions
    else:
        print("Loading model...")
        model = load_model(config, args.checkpoint_path)
        model.eval()

        sentences = load_sentences(sentences_path)
        print(f"Processing {len(sentences)} sentences from {sentences_path}")
        rows = predict_benchie(model, sentences)

        predictions_path = args.output_dir / f"diffopenie_{args.language}_explicit.txt"
        write_predictions(rows, predictions_path)
        sents_with_ext = len({r[0] for r in rows})
        print(
            f"Wrote {len(rows)} extractions across {sents_with_ext} sentences "
            f"to {predictions_path}"
        )

    report(Benchie, gold_path, predictions_path)


if __name__ == "__main__":
    main()

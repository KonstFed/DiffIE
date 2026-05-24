import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from diffopenie.evaluation.carb_eval import extract_span_text, load_model
from diffopenie.training.train_example import TrainingConfig
from diffopenie.utils import load_config, seed_everything


def load_wire_scorer(wire57_root: Path):
    scorer_path = wire57_root / "code" / "wire_scorer.py"
    if not scorer_path.exists():
        raise FileNotFoundError(f"wire_scorer.py not found at {scorer_path}")
    spec = importlib.util.spec_from_file_location("wire_scorer", scorer_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["wire_scorer"] = module
    spec.loader.exec_module(module)
    return module


def load_wire57_gold(wire57_root: Path):
    gold_path = wire57_root / "data" / "WiRe57_343-manual-oie.json"
    reference = json.load(open(gold_path))
    gold = {s["id"]: s["tuples"] for doc in reference.values() for s in doc}
    sentences = {s["id"]: s for doc in reference.values() for s in doc}
    return gold, sentences


@torch.no_grad()
def predict_wire57(model, sentences: dict) -> dict:
    predictions: dict[str, list[dict]] = {}
    for sent_id, sent_obj in tqdm(sentences.items(), desc="Predicting"):
        words = sent_obj["tokens"]
        triplets, probs = model.get_carb_prediction(words)
        preds = []
        for (sub_span, obj_span, pred_span), prob in zip(triplets, probs):
            subject = extract_span_text(words, sub_span)
            object_ = extract_span_text(words, obj_span)
            predicate = extract_span_text(words, pred_span)
            if subject and predicate and object_:
                preds.append(
                    {
                        "arg1": subject,
                        "rel": predicate,
                        "arg2": object_,
                        "extractor": "diffopenie",
                        "score": float(prob),
                    }
                )
        predictions[sent_id] = preds
    return predictions


def report(wire_scorer, gold: dict, predictions: dict, system_name: str = "diffopenie"):
    metrics, _ = wire_scorer.eval_system(gold, predictions)
    prec, rec = metrics["precision"], metrics["recall"]
    f1 = wire_scorer.f1(prec, rec)
    em_prec = (
        metrics["exactmatches_precision"][0] / metrics["exactmatches_precision"][1]
        if metrics["exactmatches_precision"][1]
        else 0.0
    )
    em_rec = (
        metrics["exactmatches_recall"][0] / metrics["exactmatches_recall"][1]
        if metrics["exactmatches_recall"][1]
        else 0.0
    )
    print(f"\nWiRe57 results for {system_name}:")
    print(f"  Precision:        {prec:.3f}")
    print(f"  Recall:           {rec:.3f}")
    print(f"  F1:               {f1:.3f}")
    print(
        f"  Matches:          {metrics['matches']} "
        f"(prec/rec of matches only: "
        f"{metrics['precision_of_matches']:.3f} / "
        f"{metrics['recall_of_matches']:.3f})"
    )
    print(
        f"  Exact-match P/R:  {em_prec:.3f} / {em_rec:.3f}  "
        f"(F1: {wire_scorer.f1(em_prec, em_rec):.3f})"
    )
    return {
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "exact_match_precision": em_prec,
        "exact_match_recall": em_rec,
        "matches": metrics["matches"],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained DiffOpenIE model on the WiRe57 benchmark"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument(
        "--wire57-root",
        type=Path,
        default=Path("benchmarks/WiRe57"),
        help="Path to the cloned WiRe57 repo (containing code/ and data/)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("wire57_output"),
        help="Where to write predictions JSON",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help="Optional pre-computed predictions JSON; if set, "
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

    wire_scorer = load_wire_scorer(args.wire57_root)
    gold, sentences = load_wire57_gold(args.wire57_root)
    print(f"Loaded {len(gold)} WiRe57 sentences.")

    if args.seeds:
        print("Loading model...")
        config = load_config(TrainingConfig, args.config)
        model = load_model(config, args.checkpoint_path)
        model.eval()
        args.output_dir.mkdir(parents=True, exist_ok=True)
        all_metrics: list[dict] = []
        for seed in args.seeds:
            seed_everything(seed)
            predictions = predict_wire57(model, sentences)
            out_path = args.output_dir / f"wire57_predictions_{seed}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(predictions, f, ensure_ascii=False, indent=2)
            metrics = report(wire_scorer, gold, predictions)
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
        print(f"Loading predictions from {args.predictions}")
        predictions = json.load(open(args.predictions))
    else:
        print("Loading model...")
        config = load_config(TrainingConfig, args.config)
        model = load_model(config, args.checkpoint_path)
        model.eval()
        predictions = predict_wire57(model, sentences)

        args.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = args.output_dir / "wire57_predictions.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)
        print(f"Wrote predictions to {out_path}")

    report(wire_scorer, gold, predictions)


if __name__ == "__main__":
    main()

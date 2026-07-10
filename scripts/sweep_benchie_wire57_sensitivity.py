"""Sweep tau/k/n sensitivity on BenchIE and WiRe57 from cached samples.

This is the rebuttal point-3 experiment: keep the CaRB-dev-selected operating
point fixed, vary one extractor parameter at a time, and score the transferred
setting on benchmarks without dev tuning.

Example:
    uv run python scripts/sweep_benchie_wire57_sensitivity.py \
        --config configs/lsoie_ex_2500/lsoie_ex_2500_config.yaml \
        --checkpoint-path configs/lsoie_ex_2500/weights.pt \
        --out-dir configs/lsoie_ex_2500/rebuttal_sensitivity \
        --max-n 512
"""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from diffopenie.evaluation.benchie_eval import (
    load_benchie_cls,
    load_sentences,
    report as report_benchie,
    write_predictions as write_benchie_predictions,
)
from diffopenie.evaluation.carb_eval import extract_span_text, load_model
from diffopenie.evaluation.wire57_eval import (
    load_wire57_gold,
    load_wire_scorer,
)
from diffopenie.models.discrete.extractors import LenientFrequencyExtractorConfig
from diffopenie.plotting.lsoie_cache_curve_plot import _to_triplet
from diffopenie.training.train_example import TrainingConfig
from diffopenie.utils import load_config

Triplet = tuple[
    tuple[int, int] | None,
    tuple[int, int] | None,
    tuple[int, int] | None,
]


def _span_to_list(span: tuple[int, int] | None) -> list[int] | None:
    return None if span is None else [int(span[0]), int(span[1])]


def _load_cache(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            row["samples"] = [_to_triplet(sample) for sample in row["samples"]]
            rows.append(row)
    return rows


@torch.no_grad()
def _sample_cache(
    model,
    items: list[dict[str, Any]],
    n: int,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    with open(out_path, "w", encoding="utf-8") as f:
        for item in tqdm(items, desc=f"Sampling {out_path.name}"):
            words = item["words"]
            triplets = model.get_triplets([words], n=n)
            samples = [
                [_span_to_list(sub), _span_to_list(obj), _span_to_list(pred)]
                for sub, obj, pred in triplets
            ]
            f.write(
                json.dumps(
                    {
                        "id": item["id"],
                        "sentence": item["sentence"],
                        "words": words,
                        "samples": samples,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def _fake_get_triplets(samples: list[Triplet]):
    def get_triplets(words_batch, *, n: int = 1, return_span_embs: bool = False):
        return samples[:n]

    return get_triplets


def _extract(rows: list[dict[str, Any]], *, n: int, topk: int, threshold: float):
    extractor = LenientFrequencyExtractorConfig(
        k=n,
        topk=topk,
        threshold=threshold,
    ).create()
    predictions = []
    for row in rows:
        words = row["words"]
        triplets, probs = extractor.get_carb_prediction(
            words,
            _fake_get_triplets(row["samples"]),
        )
        exs = []
        for (sub_span, obj_span, pred_span), prob in zip(triplets, probs, strict=True):
            subject = extract_span_text(words, sub_span)
            object_ = extract_span_text(words, obj_span)
            relation = extract_span_text(words, pred_span)
            if subject and relation and object_:
                exs.append((subject, relation, object_, float(prob)))
        predictions.append({"id": row["id"], "extractions": exs})
    return predictions


def _score_benchie(Benchie, gold_path: Path, predictions) -> dict[str, float]:
    rows: list[tuple[int, str, str, str]] = []
    for pred in predictions:
        sent_id = int(pred["id"])
        for subject, relation, object_, _prob in pred["extractions"]:
            rows.append((sent_id, subject, relation, object_))

    with tempfile.TemporaryDirectory(prefix="benchie_sensitivity_") as tmpdir:
        pred_path = Path(tmpdir) / "predictions.tsv"
        write_benchie_predictions(rows, pred_path)
        return report_benchie(Benchie, gold_path, pred_path, system_name="diffopenie")


def _score_wire57(wire_scorer, gold: dict, predictions) -> dict[str, float]:
    pred_by_id = {}
    for pred in predictions:
        pred_by_id[str(pred["id"])] = [
            {
                "arg1": subject,
                "rel": relation,
                "arg2": object_,
                "extractor": "diffopenie",
                "score": prob,
            }
            for subject, relation, object_, prob in pred["extractions"]
        ]

    metrics, _ = wire_scorer.eval_system(gold, pred_by_id)
    precision, recall = metrics["precision"], metrics["recall"]
    return {
        "precision": precision,
        "recall": recall,
        "f1": wire_scorer.f1(precision, recall),
    }


def _default_ns(max_n: int) -> list[int]:
    values = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    return [n for n in values if n <= max_n]


def _sweep_values(default: float | int, values: list[float | int]) -> list[float | int]:
    out = sorted(set(values + [default]))
    return out


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot(rows: list[dict[str, Any]], out_prefix: Path) -> None:
    benchmarks = ["BenchIE", "WiRe57"]
    sweeps = [
        ("tau", "threshold", "threshold"),
        ("k", "topk", "top-k extractions"),
        ("n", "n", "samples per sentence"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.2), sharey=False)
    for row_idx, benchmark in enumerate(benchmarks):
        bench_rows = [r for r in rows if r["benchmark"] == benchmark]
        for col_idx, (sweep_name, x_key, x_label) in enumerate(sweeps):
            ax = axes[row_idx, col_idx]
            sweep_rows = [r for r in bench_rows if r["sweep"] == sweep_name]
            sweep_rows.sort(key=lambda r: float(r[x_key]))
            xs = [r[x_key] for r in sweep_rows]
            ys = [r["f1"] for r in sweep_rows]
            ax.plot(xs, ys, marker="o", linewidth=1.8)
            if sweep_rows:
                op = sweep_rows[0][f"default_{x_key}"]
                ax.axvline(op, color="0.25", linestyle="--", linewidth=1.0)
            if col_idx == 0:
                ax.set_ylabel(f"{benchmark}\nF1")
            if row_idx == 1:
                ax.set_xlabel(x_label)
            ax.grid(True, alpha=0.25)
            if x_key == "n":
                ax.set_xscale("log", base=2)
            vals = ys or [0.0]
            pad = max(0.01, (max(vals) - min(vals)) * 0.15)
            ax.set_ylim(max(0.0, min(vals) - pad), min(1.0, max(vals) + pad))
            if row_idx == 0:
                ax.set_title(sweep_name)

    fig.tight_layout()
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        path = out_prefix.with_suffix(suffix)
        fig.savefig(path, bbox_inches="tight", dpi=300)
        print(f"Saved plot to {path}")
    plt.close(fig)


def _benchie_items(benchie_root: Path, language: str) -> tuple[list[dict[str, Any]], Path]:
    sentences_path = benchie_root / "data" / "sentences" / f"sample300_{language}.txt"
    sentences = load_sentences(sentences_path)
    if language == "en":
        gold_path = (
            benchie_root
            / "data"
            / "gold"
            / "2_annotators"
            / "benchie_gold_annotations_en.txt"
        )
    else:
        gold_path = benchie_root / "data" / "gold" / f"benchie_gold_annotations_{language}.txt"
    return [
        {"id": idx, "sentence": sentence, "words": sentence.split()}
        for idx, sentence in enumerate(sentences, start=1)
    ], gold_path


def _wire57_items(wire57_root: Path) -> tuple[list[dict[str, Any]], dict, dict]:
    gold, sentences = load_wire57_gold(wire57_root)
    items = [
        {
            "id": sent_id,
            "sentence": " ".join(sent["tokens"]),
            "words": sent["tokens"],
        }
        for sent_id, sent in sentences.items()
    ]
    return items, gold, sentences


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-n", type=int, default=None)
    parser.add_argument("--benchie-root", type=Path, default=Path("benchmarks/benchie"))
    parser.add_argument("--wire57-root", type=Path, default=Path("benchmarks/WiRe57"))
    parser.add_argument("--language", default="en", choices=["en", "de", "zh"])
    parser.add_argument("--benchie-cache", type=Path, default=None)
    parser.add_argument("--wire57-cache", type=Path, default=None)
    parser.add_argument("--taus", type=float, nargs="+", default=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8, 10])
    parser.add_argument("--ns", type=int, nargs="+", default=None)
    args = parser.parse_args()

    config = load_config(TrainingConfig, args.config)
    extractor_cfg = config.model.extractor
    default_tau = float(getattr(extractor_cfg, "threshold", 0.9))
    default_topk = int(getattr(extractor_cfg, "topk", 4))
    default_n = int(getattr(extractor_cfg, "k", 512))
    max_n = args.max_n or default_n
    if max_n < default_n:
        raise ValueError(f"--max-n must be >= operating n ({default_n})")

    out_dir = args.out_dir
    benchie_cache = args.benchie_cache or out_dir / f"cache_benchie_{max_n}.jsonl"
    wire57_cache = args.wire57_cache or out_dir / f"cache_wire57_{max_n}.jsonl"

    benchie_items, benchie_gold_path = _benchie_items(args.benchie_root, args.language)
    wire57_items, wire57_gold, _wire57_sentences = _wire57_items(args.wire57_root)

    missing_caches = [p for p in (benchie_cache, wire57_cache) if not p.exists()]
    if missing_caches:
        if args.checkpoint_path is None:
            raise FileNotFoundError(
                "Missing cache(s): "
                + ", ".join(str(p) for p in missing_caches)
                + ". Pass --checkpoint-path to create them."
            )
        print("Loading model for cache creation...")
        model = load_model(config, args.checkpoint_path)
        if not benchie_cache.exists():
            _sample_cache(model, benchie_items, max_n, benchie_cache)
        if not wire57_cache.exists():
            _sample_cache(model, wire57_items, max_n, wire57_cache)

    benchie_cache_rows = _load_cache(benchie_cache)
    wire57_cache_rows = _load_cache(wire57_cache)
    min_cached_n = min(
        min(len(row["samples"]) for row in benchie_cache_rows),
        min(len(row["samples"]) for row in wire57_cache_rows),
    )
    ns = args.ns or _default_ns(min_cached_n)
    ns = [n for n in ns if n <= min_cached_n]
    if default_n not in ns and default_n <= min_cached_n:
        ns.append(default_n)
        ns.sort()

    taus = _sweep_values(default_tau, args.taus)
    ks = _sweep_values(default_topk, args.ks)

    Benchie = load_benchie_cls(args.benchie_root)
    wire_scorer = load_wire_scorer(args.wire57_root)

    rows: list[dict[str, Any]] = []

    def eval_point(sweep: str, n: int, topk: int, tau: float) -> None:
        for benchmark, cache_rows, scorer in [
            ("BenchIE", benchie_cache_rows, "benchie"),
            ("WiRe57", wire57_cache_rows, "wire57"),
        ]:
            predictions = _extract(cache_rows, n=n, topk=topk, threshold=tau)
            if scorer == "benchie":
                metrics = _score_benchie(Benchie, benchie_gold_path, predictions)
            else:
                metrics = _score_wire57(wire_scorer, wire57_gold, predictions)
            row = {
                "benchmark": benchmark,
                "sweep": sweep,
                "n": n,
                "topk": topk,
                "threshold": tau,
                "default_n": default_n,
                "default_topk": default_topk,
                "default_threshold": default_tau,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "operating_point": n == default_n and topk == default_topk and tau == default_tau,
            }
            rows.append(row)
            print(
                f"[{benchmark:7s}] {sweep:3s} n={n:4d} k={topk:2d} "
                f"tau={tau:.2f} F1={metrics['f1']:.3f}"
            )

    for tau in taus:
        eval_point("tau", default_n, default_topk, float(tau))
    for topk in ks:
        eval_point("k", default_n, int(topk), default_tau)
    for n in ns:
        eval_point("n", int(n), default_topk, default_tau)

    json_path = out_dir / "sensitivity_rows.json"
    csv_path = out_dir / "sensitivity_rows.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": str(args.config),
                "checkpoint_path": str(args.checkpoint_path) if args.checkpoint_path else None,
                "benchie_cache": str(benchie_cache),
                "wire57_cache": str(wire57_cache),
                "rows": rows,
            },
            f,
            indent=2,
        )
    _write_csv(rows, csv_path)
    _plot(rows, out_dir / "sensitivity_curves")
    print(f"Saved rows to {json_path}")
    print(f"Saved rows to {csv_path}")


if __name__ == "__main__":
    main()

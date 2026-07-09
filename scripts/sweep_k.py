"""
TMP!!!!
Sweep the output budget k (extractor topk) — and optionally n and tau — over a
cached sample pool, scoring with the exact CaRB pipeline used by the n-study, and
dump every point to JSON.

Because k/tau are post-hoc re-clustering of an existing sample cache, this does not
re-run the model. Use it to show the tagger baseline's F1 is flat in k (its recall
is already at the oracle ceiling), i.e. the comparison isn't sensitive to the output
budget.

Usage:
    uv run python scripts/sweep_k.py \
        --config configs/lsoie_ex_tagger_2500/lsoie_ex_tagger_2500_config.yaml \
        --cache configs/lsoie_ex_tagger_2500/n_study/cache_dev_512.jsonl \
        --out configs/lsoie_ex_tagger_2500/n_study/k_sweep.json \
        --ks 1 2 3 4 6 8 10
"""

import argparse
import json
import sys
from pathlib import Path

from diffopenie.models.discrete.extractors import LenientFrequencyExtractorConfig
from diffopenie.plotting.lsoie_cache_curve_plot import (
    evaluate_with_carb_benchmark,
    load_cache,
    predict_with_extractor,
)
from diffopenie.training.train_example import TrainingConfig
from diffopenie.utils import load_config

MODES = {"CaRB": False, "CaRB (1-1)": True}  # binary flag per CaRB mode


def powers_of_two_upto(max_n: int) -> list[int]:
    ns, n = [], 1
    while n <= max_n:
        ns.append(n)
        n *= 2
    return ns


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="Output JSON path")
    ap.add_argument("--gold", type=Path, default=None,
                    help="CaRB gold TSV; defaults to trainer.carb_gold_path in config")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8, 10],
                    help="Output budgets (extractor topk) to sweep")
    ap.add_argument("--ns", type=int, nargs="+", default=None,
                    help="Sample counts to sweep (default: powers of 2 up to cache max)")
    ap.add_argument("--thresholds", type=float, nargs="+", default=None,
                    help="Lenient thresholds tau to sweep (default: config value)")
    ap.add_argument("--carb-python", dest="carb_python", type=str, default=sys.executable)
    args = ap.parse_args()

    config = load_config(TrainingConfig, args.config)
    ext = config.model.extractor
    default_tau = getattr(ext, "threshold", 0.7)
    gold_path = (args.gold or Path(config.trainer.carb_gold_path)).resolve()

    cache = load_cache(args.cache)
    max_n = min(len(row["samples"]) for row in cache)
    ns = args.ns or powers_of_two_upto(max_n)
    ns = [n for n in ns if n <= max_n]
    taus = args.thresholds or [default_tau]

    print(f"{len(cache)} sentences | max n={max_n}")
    print(f"grid: modes={list(MODES)} ns={ns} ks={args.ks} taus={taus}")

    rows: list[dict] = []
    for mode, binary in MODES.items():
        for tau in taus:
            for n in ns:
                for k in args.ks:
                    extractor = LenientFrequencyExtractorConfig(
                        k=n, topk=k, threshold=tau
                    ).create()
                    pred = predict_with_extractor(cache, extractor, n)
                    res = evaluate_with_carb_benchmark(
                        gold_path, pred, carb_python=args.carb_python, binary=binary
                    )
                    rows.append({
                        "mode": mode, "n": n, "topk": k, "threshold": tau,
                        "f1": res.f1, "precision": res.precision,
                        "recall": res.recall, "auc": res.auc,
                    })
                    print(f"[{mode}] n={n:4d} k={k:2d} tau={tau:.2f}  "
                          f"F1={res.f1:.3f} P={res.precision:.3f} R={res.recall:.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({
            "config": str(args.config),
            "cache": str(args.cache),
            "gold": str(gold_path),
            "default_threshold": default_tau,
            "rows": rows,
        }, f, indent=2)
    print(f"\nWrote {len(rows)} rows to {args.out}")

    # Best-F1 per mode, for a quick read of whether k matters.
    print("\n--- best F1 per mode ---")
    for mode in MODES:
        best = max((r for r in rows if r["mode"] == mode), key=lambda r: r["f1"])
        print(f"  {mode:12s} F1={best['f1']:.3f} at n={best['n']} k={best['topk']} "
              f"tau={best['threshold']:.2f} (P={best['precision']:.3f} R={best['recall']:.3f})")


if __name__ == "__main__":
    main()

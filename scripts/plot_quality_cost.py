"""Plot a quality-cost curve: CaRB F1 vs inference cost per sentence, one line per
model (e.g. DiffIE vs the MC-dropout tagger), with each point labelled by its
sample count n.

Each --series takes: LABEL  TIMING_CSV  F1_SOURCE
  TIMING_CSV : output of diffopenie.evaluation.timing_benchmark (must have
               columns n_samples and <--x>; averaged over repeats per n).
  F1_SOURCE  : either
                 - an nkt_sweep.json (rows with n/topk/threshold/mode/f1) — the
                   --mode/--topk/--threshold filters pick one F1 per n, or
                 - a CSV with an n column (n or n_samples) and an f1 column
                   (f1 or lenient_f1).

Example:
    uv run python scripts/plot_quality_cost.py \
        --series DiffIE configs/lsoie_ex_2500/timing_diffie.csv \
                 configs/lsoie_ex_2500/n_study/curve.csv \
        --series "MC-dropout tagger" \
                 configs/lsoie_ex_tagger_2500/timing_tagger.csv nkt_sweep.json \
        --mode CaRB --topk 4 --threshold 0.9 \
        --x total_sec_per_sentence \
        --out configs/quality_cost.png
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def _load_cost(timing_csv: Path, x_col: str) -> pd.Series:
    df = pd.read_csv(timing_csv)
    if "n_samples" not in df.columns:
        raise ValueError(f"{timing_csv} has no 'n_samples' column")
    if x_col not in df.columns:
        raise ValueError(f"{timing_csv} has no '{x_col}' column; have {list(df.columns)}")
    return df.groupby("n_samples")[x_col].mean().sort_index()


def _load_f1(f1_source: Path, mode: str, topk: int, threshold: float) -> pd.Series:
    if f1_source.suffix == ".json":
        rows = json.loads(f1_source.read_text())["rows"]
        sel = [
            r for r in rows
            if r.get("mode") == mode
            and int(r["topk"]) == topk
            and abs(float(r["threshold"]) - threshold) < 1e-9
        ]
        if not sel:
            raise ValueError(
                f"No rows in {f1_source} for mode={mode} topk={topk} "
                f"threshold={threshold}. Check --mode/--topk/--threshold."
            )
        s = pd.Series({int(r["n"]): float(r["f1"]) for r in sel})
        return s.sort_index()

    df = pd.read_csv(f1_source)
    n_col = next((c for c in ("n", "n_samples") if c in df.columns), None)
    f1_col = next((c for c in ("f1", "lenient_f1") if c in df.columns), None)
    if n_col is None or f1_col is None:
        raise ValueError(
            f"{f1_source}: need an n column (n/n_samples) and an f1 column "
            f"(f1/lenient_f1); have {list(df.columns)}"
        )
    return df.groupby(n_col)[f1_col].mean().sort_index()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--series", action="append", nargs=3, required=True,
                    metavar=("LABEL", "TIMING_CSV", "F1_SOURCE"),
                    help="Repeatable. One model per --series.")
    ap.add_argument("--x", default="total_sec_per_sentence",
                    help="Timing column for the x-axis (cost). "
                         "e.g. total_sec_per_sentence, model_sec_per_sentence")
    ap.add_argument("--mode", default="CaRB", help="CaRB mode for JSON F1 sources")
    ap.add_argument("--topk", type=int, default=4, help="topk filter for JSON F1 sources")
    ap.add_argument("--threshold", type=float, default=0.9,
                    help="tau filter for JSON F1 sources")
    ap.add_argument("--out", type=Path, default=Path("quality_cost.png"))
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    merged_out = []
    for label, timing_csv, f1_source in args.series:
        cost = _load_cost(Path(timing_csv), args.x)
        f1 = _load_f1(Path(f1_source), args.mode, args.topk, args.threshold)
        common = sorted(set(cost.index) & set(f1.index))
        if not common:
            raise ValueError(f"[{label}] no shared n between timing and F1 sources")
        xs = [cost[n] for n in common]
        ys = [f1[n] for n in common]
        ax.plot(xs, ys, marker="o", markersize=4, label=label)
        for n, x, y in zip(common, xs, ys):
            ax.annotate(f"n={n}", (x, y), fontsize=7,
                        textcoords="offset points", xytext=(4, 4))
        for n, x, y in zip(common, xs, ys):
            merged_out.append({"series": label, "n": n, "cost": x, "f1": y})

    ax.set_xscale("log")
    ax.set_xlabel(f"inference cost — {args.x} (s, log scale)")
    ax.set_ylabel(f"{args.mode} F1")
    ax.set_title(args.title or f"Quality vs cost ({args.mode})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.savefig(args.out.with_suffix(".pdf"), bbox_inches="tight")
    csv_out = args.out.with_suffix(".csv")
    pd.DataFrame(merged_out).to_csv(csv_out, index=False)
    print(f"Wrote {args.out}, {args.out.with_suffix('.pdf')}, {csv_out}")


if __name__ == "__main__":
    main()

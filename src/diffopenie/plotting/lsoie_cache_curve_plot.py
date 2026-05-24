"""Plot dev precision/recall/F1 curves from cached diffusion samples.

This script sweeps the number of cached samples `n` from 1 to the maximum
available per sentence, evaluates two extractors on CaRB dev:

* lenient frequency extractor
* exact frequency extractor

Both use the same `topk` from the provided training config. The lenient
extractor also reuses its configured overlap threshold.

The output is a paper-friendly 1x3 figure with shared x-axis, one panel each
for precision, recall, and F1.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FixedFormatter, FixedLocator, LogLocator, MaxNLocator, NullFormatter, NullLocator
from sklearn.metrics import auc as sk_auc
from tqdm import tqdm

from diffopenie.evaluation.carb_metrics import Extraction, load_gold_file
from diffopenie.models.discrete.extractors import (
    FrequencyExtractorConfig,
    LenientFrequencyExtractorConfig,
)
from diffopenie.training.train_example import TrainingConfig
from diffopenie.utils import load_config

Triplet = tuple[
    tuple[int, int] | None,
    tuple[int, int] | None,
    tuple[int, int] | None,
]


@dataclass
class CarbResult:
    auc: float
    precision: float
    recall: float
    f1: float
    oracle_recall: float = 0.0


def _to_triplet(raw: list) -> Triplet:
    def span(x):
        return None if x is None else (int(x[0]), int(x[1]))

    return (span(raw[0]), span(raw[1]), span(raw[2]))


def load_cache(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            row = json.loads(ln)
            row["samples"] = [_to_triplet(sample) for sample in row["samples"]]
            rows.append(row)
    return rows


def _extract_text(words: list[str], span: tuple[int, int] | None) -> str:
    if span is None:
        return ""
    return " ".join(words[span[0] : span[1] + 1])


def _to_extraction(words: list[str], triplet: Triplet, confidence: float) -> Extraction:
    sub, obj, pred = triplet
    subj = _extract_text(words, sub)
    obj_ = _extract_text(words, obj)
    pred_ = _extract_text(words, pred)
    return Extraction(pred=pred_, args=[subj, obj_], confidence=confidence)


def write_tsv(predictions: dict[str, list[Extraction]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for sent, exs in predictions.items():
            for ex in exs:
                args_str = "\t".join(ex.args)
                f.write(f"{sent}\t{ex.confidence}\t{ex.pred}\t{args_str}\n")


def _make_fake_get_triplets(samples: list[Triplet]):
    def fake_get_triplets(words_batch, *, n, return_span_embs=False):
        return samples[:n]

    return fake_get_triplets


def predict_with_extractor(
    cache: list[dict],
    extractor,
    n: int,
) -> dict[str, list[Extraction]]:
    predicted: dict[str, list[Extraction]] = {}
    for row in cache:
        words = row["words"]
        samples = row["samples"]
        if n > len(samples):
            raise ValueError(
                f"requested n={n} > cached samples={len(samples)} for one sentence"
            )

        triplets, probs = extractor.get_carb_prediction(
            words, _make_fake_get_triplets(samples)
        )
        extractions = [
            _to_extraction(words, triplet, float(prob))
            for triplet, prob in zip(triplets, probs)
            if all(span is not None for span in triplet)
        ]
        if extractions:
            predicted[row["sentence"]] = extractions
    return predicted


def _resolve_carb_root() -> Path:
    return Path(__file__).resolve().parents[3] / "benchmarks" / "CaRB"


def evaluate_with_carb_benchmark(
    gold_path: Path,
    predicted: dict[str, list[Extraction]],
    *,
    carb_python: str,
    binary: bool,
) -> CarbResult:
    carb_root = _resolve_carb_root()
    if not carb_root.exists():
        raise FileNotFoundError(f"CaRB benchmark directory not found at {carb_root}")
    gold_path = gold_path.resolve()

    with tempfile.TemporaryDirectory(prefix="carb_curve_") as tmpdir:
        tmpdir_p = Path(tmpdir)
        pred_path = tmpdir_p / "predictions.tsv"
        out_path = tmpdir_p / "curve.dat"
        write_tsv(predicted, pred_path)

        cmd = [
            carb_python,
            "carb.py",
            "--gold",
            str(gold_path),
            "--out",
            str(out_path),
            "--tabbed",
            str(pred_path),
        ]
        if binary:
            cmd.append("--binary")

        proc = subprocess.run(
            cmd,
            cwd=carb_root,
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                "CaRB benchmark failed\n"
                f"cmd: {' '.join(cmd)}\n"
                f"stdout:\n{proc.stdout}\n"
                f"stderr:\n{proc.stderr}"
            )

        rows: list[tuple[float, float]] = []
        with open(out_path, "r", encoding="utf-8") as f:
            next(f, None)  # header
            for line in f:
                if not line.strip():
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 2:
                    continue
                rows.append((float(parts[0]), float(parts[1])))

        if not rows:
            return CarbResult(auc=0.0, precision=0.0, recall=0.0, f1=0.0)

        rows.sort(key=lambda x: x[1])

        precisions = [p for p, _r in rows]
        recalls = [r for _p, r in rows]
        f1s = [
            2 * p * r / (p + r) if (p + r) > 0 else 0.0
            for p, r in zip(precisions, recalls, strict=True)
        ]
        opt_idx = max(range(len(f1s)), key=lambda i: f1s[i])

        temp_rec = [0.0] + recalls
        temp_prec = [1.0] + precisions
        return CarbResult(
            auc=round(float(sk_auc(temp_rec, temp_prec)), 3),
            precision=round(precisions[opt_idx], 3),
            recall=round(recalls[opt_idx], 3),
            f1=round(f1s[opt_idx], 3),
        )


def _metric_rows(
    cache: list[dict],
    gold_path: Path,
    lenient_cfg: LenientFrequencyExtractorConfig,
    freq_cfg: FrequencyExtractorConfig,
    sweep_ns: list[int],
    *,
    mode_label: str,
    carb_python: str,
    binary: bool,
) -> list[dict]:
    rows: list[dict] = []

    for n in tqdm(sweep_ns, desc=f"Sweeping n [{mode_label}]"):
        lenient_extractor = LenientFrequencyExtractorConfig(
            k=n,
            topk=lenient_cfg.topk,
            threshold=lenient_cfg.threshold,
        ).create()
        frequency_extractor = FrequencyExtractorConfig(
            k=n,
            topk=freq_cfg.topk,
        ).create()

        lenient_pred = predict_with_extractor(cache, lenient_extractor, n)
        freq_pred = predict_with_extractor(cache, frequency_extractor, n)

        lenient_result = evaluate_with_carb_benchmark(
            gold_path,
            lenient_pred,
            carb_python=carb_python,
            binary=binary,
        )
        freq_result = evaluate_with_carb_benchmark(
            gold_path,
            freq_pred,
            carb_python=carb_python,
            binary=binary,
        )

        rows.append(
            {
                "n": n,
                "lenient_precision": lenient_result.precision,
                "lenient_recall": lenient_result.recall,
                "lenient_f1": lenient_result.f1,
                "frequency_precision": freq_result.precision,
                "frequency_recall": freq_result.recall,
                "frequency_f1": freq_result.f1,
                "lenient_auc": lenient_result.auc,
                "frequency_auc": freq_result.auc,
                "lenient_oracle_recall": lenient_result.oracle_recall,
                "frequency_oracle_recall": freq_result.oracle_recall,
            }
        )

        print(
            f"[{mode_label}] n={n:4d}  "
            f"lenient F1={lenient_result.f1:.3f} P={lenient_result.precision:.3f} "
            f"R={lenient_result.recall:.3f}  "
            f"freq F1={freq_result.f1:.3f} P={freq_result.precision:.3f} "
            f"R={freq_result.recall:.3f}"
        )

    return rows


def _metric_rows_by_mode(
    cache: list[dict],
    gold_path: Path,
    lenient_cfg: LenientFrequencyExtractorConfig,
    freq_cfg: FrequencyExtractorConfig,
    sweep_ns: list[int],
    *,
    carb_python: str,
) -> dict[str, list[dict]]:
    return {
        "CaRB": _metric_rows(
            cache,
            gold_path,
            lenient_cfg,
            freq_cfg,
            sweep_ns,
            mode_label="CaRB",
            carb_python=carb_python,
            binary=False,
        ),
        "CaRB (1-1)": _metric_rows(
            cache,
            gold_path,
            lenient_cfg,
            freq_cfg,
            sweep_ns,
            mode_label="CaRB (1-1)",
            carb_python=carb_python,
            binary=True,
        ),
    }


def _save_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_n_schedule(max_n: int) -> list[int]:
    """Hardcoded schedule: 1-step to 30, 5-step to 100, 10-step to 1000."""
    if max_n <= 0:
        return []

    ns: list[int] = list(range(1, min(max_n, 30) + 1))
    if max_n > 30:
        ns.extend(range(35, min(max_n, 100) + 1, 5))
    if max_n > 100:
        ns.extend(range(110, min(max_n, 1000) + 1, 10))
    if not ns or ns[-1] != max_n:
        ns.append(max_n)

    seen: set[int] = set()
    ordered: list[int] = []
    for n in ns:
        if n not in seen:
            ordered.append(n)
            seen.add(n)
    return ordered


def _configure_matplotlib() -> None:
    sns.set_theme(
        style="whitegrid",
        font="serif",
        rc={
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        },
    )


def _mode_to_df(rows: list[dict]) -> "pd.DataFrame":
    import pandas as pd
    records = []
    for row in rows:
        for ext, prefix in [("Lenient", "lenient"), ("Frequency", "frequency")]:
            records.append({
                "n": row["n"],
                "Extractor": ext,
                "Precision": row[f"{prefix}_precision"],
                "Recall": row[f"{prefix}_recall"],
                "F1": row[f"{prefix}_f1"],
            })
    return pd.DataFrame(records)


def plot_joint_curves(
    mode_rows: dict[str, list[dict]],
    out_prefix: Path,
    title: str | None = None,
) -> None:
    _configure_matplotlib()
    palette = {"Lenient": sns.color_palette()[0], "Frequency": sns.color_palette()[1]}
    dashes  = {"Lenient": "",  "Frequency": (4, 2)}
    markers = {"Lenient": "o", "Frequency": "s"}

    mode_order = ["CaRB", "CaRB (1-1)"]
    metrics = [("Precision", "Precision"), ("Recall", "Recall"), ("F1", "F1")]

    fig, axes = plt.subplots(2, 3, figsize=(14.5, 7.6), sharex=True, sharey=False)

    for row_idx, mode in enumerate(mode_order):
        df = _mode_to_df(mode_rows[mode])
        ns = sorted(df["n"].unique().tolist())

        for col_idx, (col, label) in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            sns.lineplot(
                data=df, x="n", y=col, hue="Extractor", style="Extractor",
                dashes=dashes, markers=markers,
                palette=palette, markersize=7,
                markeredgecolor="white", markeredgewidth=1.1,
                linewidth=1.8, legend=(row_idx == 0 and col_idx == 0),
                ax=ax,
            )
            if row_idx == 0:
                ax.set_title(label)
            ax.set_xlabel("")
            ax.set_ylabel("Score" if col_idx == 0 else "")
            vals = df[col].tolist()
            lo, hi = min(vals), max(vals)
            pad = max(0.02, (hi - lo) * 0.12)
            ax.set_ylim(max(0.0, lo - pad), min(1.0, hi + pad))
            _apply_log2_xaxis(ax, ns)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
            # restore whitegrid x-lines that log scale may have cleared
            ax.grid(True, which="minor", axis="x", alpha=0.15, linewidth=0.5,
                    color="lightgrey", zorder=0)

        axes[row_idx, 0].annotate(
            mode, xy=(0.0, 1.10), xycoords="axes fraction",
            ha="left", va="bottom", fontsize=13, fontweight="semibold",
        )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.02))
    fig.supxlabel("number of samples per sentence")
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))

    png_path = out_prefix.with_suffix(".png")
    pdf_path = out_prefix.with_suffix(".pdf")
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {png_path}")
    print(f"Saved figure to {pdf_path}")


_LOG2_NS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]


def _apply_log2_xaxis(ax, ns: list[int]) -> None:
    """Log-2 x-axis: major ticks pinned to ns, minor ticks show log spacing."""
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator(ns))
    ax.xaxis.set_major_formatter(FixedFormatter([str(n) for n in ns]))
    # minor ticks at 3 intermediate positions per octave — gives the
    # characteristic bunching that makes a log scale visually recognisable
    ax.xaxis.set_minor_locator(LogLocator(base=2, subs=np.arange(1.25, 2.0, 0.25)))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(ns[0] / 1.4, ns[-1] * 1.4)


def plot_f1_curves(
    mode_rows: dict[str, list[dict]],
    out_prefix: Path,
) -> None:
    """2-row × 1-col, F1 only, seaborn + log-2 x-axis."""
    _configure_matplotlib()
    palette = {"Lenient": sns.color_palette()[0], "Frequency": sns.color_palette()[1]}
    dashes  = {"Lenient": "",  "Frequency": (4, 2)}
    markers = {"Lenient": "o", "Frequency": "s"}

    mode_order = ["CaRB", "CaRB (1-1)"]
    fig, axes = plt.subplots(2, 1, figsize=(5, 6), sharex=True)

    for row_idx, mode in enumerate(mode_order):
        df = _mode_to_df(mode_rows[mode])
        ns = sorted(df["n"].unique().tolist())
        ax = axes[row_idx]

        sns.lineplot(
            data=df, x="n", y="F1", hue="Extractor", style="Extractor",
            dashes=dashes, markers=markers,
            palette=palette, markersize=7,
            markeredgecolor="white", markeredgewidth=1.1,
            linewidth=1.8, legend=(row_idx == 0),
            ax=ax,
        )
        vals = df["F1"].tolist()
        lo, hi = min(vals), max(vals)
        pad = max(0.02, (hi - lo) * 0.12)
        ax.set_ylim(max(0.0, lo - pad), min(1.0, hi + pad))
        ax.set_xlabel("")
        ax.set_ylabel("F1")
        _apply_log2_xaxis(ax, ns)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.grid(True, which="minor", axis="x", alpha=0.15, linewidth=0.5,
                color="lightgrey", zorder=0)
        ax.annotate(
            mode, xy=(0.0, 1.04), xycoords="axes fraction",
            ha="left", va="bottom", fontsize=11, fontweight="semibold",
        )

    axes[0].legend(loc="lower right", frameon=False)
    axes[-1].set_xlabel("number of samples per sentence")
    fig.tight_layout()

    out_prefix = out_prefix.parent / (out_prefix.stem + "_f1")
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        path = out_prefix.with_suffix(suffix)
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved figure to {path}")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Sweep cache size and plot dev precision/recall/F1 curves"
    )
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument(
        "--gold",
        type=Path,
        default=None,
        help="CaRB gold TSV file. Defaults to trainer.carb_gold_path from config.",
    )
    ap.add_argument(
        "--out-prefix",
        type=Path,
        default=Path("plotting/lsoie_cache_curve"),
        help="Output path prefix for .png/.pdf/.csv",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap the sampled-n sweep at N (for example, 30)",
    )
    ap.add_argument(
        "--title",
        type=str,
        default="CaRB metrics curves vs cached sample count",
    )
    ap.add_argument(
        "-carb-python",
        "--carb-python",
        dest="carb_python",
        type=str,
        default=sys.executable,
        help="Python executable used to run benchmarks/CaRB/carb.py",
    )
    args = ap.parse_args()

    config = load_config(TrainingConfig, args.config)
    extractor_cfg = config.model.extractor
    if not isinstance(extractor_cfg, LenientFrequencyExtractorConfig):
        print(
            "[warn] config.model.extractor is not lenient_frequency; "
            "using its topk if available, but comparing lenient vs exact frequency."
        )

    topk = getattr(extractor_cfg, "topk", None)
    threshold = getattr(extractor_cfg, "threshold", 0.7)
    if topk is None:
        raise ValueError("config.model.extractor.topk is required for this plot")

    gold_path = args.gold or Path(config.trainer.carb_gold_path)
    gold_path = gold_path.resolve()
    print(f"Loading cache from {args.cache}...")
    cache = load_cache(args.cache)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be a positive integer")
        print(f"  limiting sweep to n <= {args.limit}")
    max_cache_n = min(len(row["samples"]) for row in cache)
    max_n = min(max_cache_n, args.limit) if args.limit is not None else max_cache_n
    print(f"  {len(cache)} sentences, max valid n = {max_cache_n}")
    print(f"  sweep max n = {max_n}")
    sweep_ns = [n for n in _LOG2_NS if n <= max_n]
    print(f"  sweep points: {len(sweep_ns)} — {sweep_ns}")

    print(f"Loading gold from {gold_path}...")
    gold = load_gold_file(str(gold_path))
    print(
        f"  {len(gold)} gold sentences, "
        f"{sum(len(v) for v in gold.values())} gold extractions"
    )

    lenient_cfg = LenientFrequencyExtractorConfig(
        k=max_n,
        topk=topk,
        threshold=threshold,
    )
    freq_cfg = FrequencyExtractorConfig(k=max_n, topk=topk)

    mode_rows = _metric_rows_by_mode(
        cache,
        gold_path,
        lenient_cfg,
        freq_cfg,
        sweep_ns,
        carb_python=args.carb_python,
    )

    csv_path = args.out_prefix.with_suffix(".csv")
    joint_rows: list[dict] = []
    for mode, rows in mode_rows.items():
        for row in rows:
            joint_rows.append({"mode": mode, **row})
    _save_csv(joint_rows, csv_path)
    print(f"Saved curve data to {csv_path}")

    plot_joint_curves(mode_rows, args.out_prefix, title=args.title)
    plot_f1_curves(mode_rows, args.out_prefix)


if __name__ == "__main__":
    main()

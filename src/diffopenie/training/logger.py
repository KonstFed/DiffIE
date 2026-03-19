"""Training logger: CSV persistence, console output, plot generation."""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from matplotlib.gridspec import GridSpec

# Percentiles for confidence band (e.g. 25–75% band)
PER_T_BAND_LOW, PER_T_BAND_HIGH = 25.0, 75.0

from diffopenie.training.metrics import (
    CLASS_NAMES,
    MetricsResult,
    PerTimestepMetricsResult,
)


class TrainingLogger:
    """Handles CSV logging, console output, and plot generation."""

    def __init__(self, log_path: Path | None):
        self.log_path = log_path
        self._rows: list[dict] = []

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float | None = None,
        direct_metrics: MetricsResult | None = None,
        carb_metrics: MetricsResult | None = None,
        train_carb_metrics: MetricsResult | None = None,
        per_t_loss: torch.Tensor | None = None,
        per_t_val_loss: torch.Tensor | None = None,
        t_sampled_counts: torch.Tensor | None = None,
        per_t_carb_metrics: PerTimestepMetricsResult | None = None,
        train_per_t_carb_metrics: PerTimestepMetricsResult | None = None,
        carb_result=None,
    ):
        row: dict = {"epoch": epoch, "train_loss": train_loss}
        if val_loss is not None:
            row["val_loss"] = val_loss
        if direct_metrics is not None:
            row.update(direct_metrics.to_dict("direct_"))
        if carb_metrics is not None:
            row.update(carb_metrics.to_dict(""))
        if train_carb_metrics is not None:
            row.update(train_carb_metrics.to_dict("train_"))
        if carb_result is not None:
            row.update(carb_result.to_dict(""))
        self._rows.append(row)

        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self._rows).to_csv(self.log_path, index=False)
        self._plot_training()
        if per_t_loss is not None:
            self._plot_per_t_loss(per_t_loss, epoch, per_t_val_loss, t_sampled_counts)
        if per_t_carb_metrics is not None or train_per_t_carb_metrics is not None:
            self._plot_per_t_carb_merged(
                per_t_carb_metrics, train_per_t_carb_metrics, epoch
            )

    def print_epoch(
        self,
        epoch: int,
        num_epochs: int,
        train_loss: float,
        val_loss: float | None,
        direct_metrics: MetricsResult | None,
        carb_metrics: MetricsResult | None,
        train_carb_metrics: MetricsResult | None,
        best_f1: float | None = None,
        new_best: float | None = None,
        carb_result=None,
    ):
        C, B, D, M, G, R = (
            "\033[36m", "\033[1m", "\033[2m", "\033[35m", "\033[32m", "\033[0m",
        )
        print(f"\n{B}{C}Epoch {epoch}/{num_epochs}{R}")
        print(f"  {D}train_loss{R}\t{train_loss:.4g}")
        if val_loss is not None:
            print(f"  {D}val_loss{R}\t{val_loss:.4g}")
        if direct_metrics:
            print(
                f"  {D}direct    "
                f"P={direct_metrics.precision:.3f}  "
                f"R={direct_metrics.recall:.3f}  "
                f"F1={direct_metrics.f1:.3f}{R}"
            )
        if train_carb_metrics:
            print(
                f"  {D}train_carb "
                f"P={train_carb_metrics.precision:.3f}  "
                f"R={train_carb_metrics.recall:.3f}  "
                f"F1={train_carb_metrics.f1:.3f}{R}"
            )
        if carb_metrics:
            print(
                f"  {B}{M}val_carb  "
                f"P={carb_metrics.precision:.3f}  "
                f"R={carb_metrics.recall:.3f}  "
                f"F1={carb_metrics.f1:.3f}{R}"
            )
        if carb_result:
            print(
                f"  {B}{M}CaRB     "
                f"AUC={carb_result.auc:.3f}  "
                f"P={carb_result.precision:.3f}  "
                f"R={carb_result.recall:.3f}  "
                f"F1={carb_result.f1:.3f}{R}"
            )
        # Support both old 'best_f1' and new 'new_best' parameter names
        _best = new_best if new_best is not None else best_f1
        if _best is not None:
            print(f"  {G}New best F1: {_best:.4f}{R}")

    # -- Plotting ---------------------------------------------------------

    def _plot_training(self):
        if self.log_path is None:
            return
        df = pd.read_csv(self.log_path)
        if df.empty or "epoch" not in df.columns:
            return

        plot_path = self.log_path.parent / f"{self.log_path.stem}_plots.png"

        # Determine which metric sections have data
        has_lsoie = any(c in df.columns for c in ("precision", "recall", "f1"))
        has_carb = any(c in df.columns for c in ("carb_auc", "carb_f1"))
        n_rows = 2 + int(has_lsoie) + int(has_carb)

        fig = plt.figure(figsize=(12, 3.5 * n_rows))
        gs = GridSpec(n_rows, 1, figure=fig)
        epochs = df["epoch"]
        row = 0

        ax1 = fig.add_subplot(gs[row])
        _plot_cols(ax1, df, epochs, [
            ("train_loss", "Train"),
            ("val_loss", "Val"),
        ])
        ax1.set_title("Loss")
        ax1.grid(True, alpha=0.3)
        row += 1

        ax2 = fig.add_subplot(gs[row], sharex=ax1)
        _plot_cols(ax2, df, epochs, [
            ("direct_precision", "P"),
            ("direct_recall", "R"),
            ("direct_f1", "F1"),
        ])
        ax2.set_title("Direct metrics (train forward pass)")
        ax2.grid(True, alpha=0.3)
        row += 1

        if has_lsoie:
            ax_lsoie = fig.add_subplot(gs[row], sharex=ax1)
            _plot_cols(ax_lsoie, df, epochs, [
                ("precision", "Val P"),
                ("recall", "Val R"),
                ("f1", "Val F1"),
                ("train_precision", "Train P"),
                ("train_recall", "Train R"),
                ("train_f1", "Train F1"),
            ])
            ax_lsoie.set_title("LSOIE token-overlap validation")
            ax_lsoie.set_ylim(0, 1.05)
            ax_lsoie.grid(True, alpha=0.3)
            row += 1

        if has_carb:
            ax_carb = fig.add_subplot(gs[row], sharex=ax1)
            _plot_cols(ax_carb, df, epochs, [
                ("carb_auc", "AUC"),
                ("carb_precision", "P"),
                ("carb_recall", "R"),
                ("carb_f1", "F1"),
            ])
            ax_carb.set_title("CaRB benchmark (dev)")
            ax_carb.set_ylim(0, 1.05)
            ax_carb.grid(True, alpha=0.3)
            row += 1

        axes = fig.get_axes()
        if axes:
            axes[-1].set_xlabel("Epoch")

        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()

    def _plot_per_t_loss(
        self,
        per_t_loss: torch.Tensor,
        epoch: int,
        per_t_val_loss: torch.Tensor | None = None,
        t_sampled_counts: torch.Tensor | None = None,
    ):
        if self.log_path is None:
            return
        plot_path = self.log_path.parent / "per_t_loss.png"
        t_vals = torch.arange(1, len(per_t_loss) + 1).numpy()
        train_vals = per_t_loss.cpu().numpy()

        fig, ax = plt.subplots(figsize=(8, 4))
        # Background: distribution of t sampled in this epoch (twin axis)
        if t_sampled_counts is not None and len(t_sampled_counts) == len(per_t_loss):
            ax_twin = ax.twinx()
            counts = t_sampled_counts.cpu().numpy().astype(float)
            total = counts.sum()
            if total > 0:
                frac = counts / total
                ax_twin.bar(
                    t_vals, frac, width=0.8, alpha=0.3, color="gray",
                    align="center", label="t sampled",
                )
                ax_twin.set_ylabel("Fraction of samples (t)", color="gray", fontsize=9)
                ax_twin.tick_params(axis="y", labelcolor="gray", labelsize=8)
                ax_twin.set_ylim(0, None)
        ax.plot(
            t_vals, train_vals, label="Train", marker="o", markersize=3,
        )
        if per_t_val_loss is not None and len(per_t_val_loss) == len(per_t_loss):
            val_vals = per_t_val_loss.cpu().numpy()
            ax.plot(
                t_vals, val_vals, label="Val", marker="s", markersize=3,
            )
        ax.set_xlabel("Timestep t")
        ax.set_ylabel("Avg loss")
        ax.set_title(f"Per-timestep loss (epoch {epoch})")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()

    def _plot_per_t_carb_merged(
        self,
        per_t_carb: PerTimestepMetricsResult | None,
        train_per_t_carb: PerTimestepMetricsResult | None,
        epoch: int,
    ):
        """Plot val and train CaRB metrics per timestep with confidence bands (percentile over sequences)."""
        if self.log_path is None:
            return
        plot_path = self.log_path.parent / "carb_per_t.png"
        n_plots = sum(x is not None for x in (per_t_carb, train_per_t_carb))
        if n_plots == 0:
            return
        fig, axes = plt.subplots(
            1, 2, figsize=(10, 4), sharex=True, sharey=True
        )
        for ax, data, kind in zip(
            axes,
            (per_t_carb, train_per_t_carb),
            ("Val", "Train"),
        ):
            if data is None:
                ax.set_visible(False)
                continue
            T = data.f1.shape[0]
            t_vals = np.array(list(range(T, 0, -1)))
            band_alpha = 0.25
            # Confidence band from per-sequence percentiles
            if data.per_sequence_f1 is not None:
                arr = data.per_sequence_f1.numpy()
                lo = np.percentile(arr, PER_T_BAND_LOW, axis=0)
                hi = np.percentile(arr, PER_T_BAND_HIGH, axis=0)
                ax.fill_between(t_vals, lo, hi, color="C2", alpha=band_alpha, zorder=1)
            if data.per_sequence_precision is not None:
                arr = data.per_sequence_precision.numpy()
                lo = np.percentile(arr, PER_T_BAND_LOW, axis=0)
                hi = np.percentile(arr, PER_T_BAND_HIGH, axis=0)
                ax.fill_between(t_vals, lo, hi, color="C0", alpha=band_alpha, zorder=1)
            if data.per_sequence_recall is not None:
                arr = data.per_sequence_recall.numpy()
                lo = np.percentile(arr, PER_T_BAND_LOW, axis=0)
                hi = np.percentile(arr, PER_T_BAND_HIGH, axis=0)
                ax.fill_between(t_vals, lo, hi, color="C1", alpha=band_alpha, zorder=1)
            if data.per_sequence_ratio_masked is not None:
                arr = data.per_sequence_ratio_masked.numpy()
                lo = np.percentile(arr, PER_T_BAND_LOW, axis=0)
                hi = np.percentile(arr, PER_T_BAND_HIGH, axis=0)
                ax.fill_between(t_vals, lo, hi, color="C3", alpha=band_alpha, zorder=1)
            # Usual aggregate metrics (on top)
            p = data.precision.cpu().numpy()
            r = data.recall.cpu().numpy()
            f = data.f1.cpu().numpy()
            ax.plot(t_vals, p, label="Precision", marker="o", markersize=3, color="C0", zorder=10)
            ax.plot(t_vals, r, label="Recall", marker="s", markersize=3, color="C1", zorder=10)
            ax.plot(t_vals, f, label="F1", marker="^", markersize=3, color="C2", zorder=10)
            if data.ratio_masked is not None:
                rm = data.ratio_masked.cpu().numpy()
                ax.plot(
                    t_vals, rm, label="Ratio masked", marker="d", markersize=3, color="C3", zorder=10,
                )
            ax.set_xlabel("Timestep t")
            ax.set_ylabel("Score")
            ax.set_title(f"{kind} (epoch {epoch})")
            ax.set_ylim(0, 1.05)
            ax.legend()
            ax.grid(True, alpha=0.3)
        fig.suptitle("CaRB metrics per timestep", fontsize=12, y=1.02)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()


def _plot_cols(ax, df, epochs, cols: list[tuple[str, str]]):
    for col, label in cols:
        if col not in df.columns:
            continue
        valid = df[col].notna()
        if valid.any():
            ax.plot(
                df.loc[valid, "epoch"], df.loc[valid, col],
                label=label, marker="o", markersize=3,
            )
    if ax.get_legend_handles_labels()[1]:
        ax.legend(fontsize=7)

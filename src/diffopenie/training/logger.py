"""Training logger: CSV persistence, console output, plot generation."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from matplotlib.gridspec import GridSpec

from diffopenie.training.metrics import CLASS_NAMES, MetricsResult


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
        self._rows.append(row)

        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self._rows).to_csv(self.log_path, index=False)
        self._plot_training()
        if per_t_loss is not None:
            self._plot_per_t_loss(per_t_loss, epoch)

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
        if best_f1 is not None:
            print(f"  {G}New best F1: {best_f1:.4f}{R}")

    # -- Plotting ---------------------------------------------------------

    def _plot_training(self):
        if self.log_path is None:
            return
        df = pd.read_csv(self.log_path)
        if df.empty or "epoch" not in df.columns:
            return

        plot_path = self.log_path.parent / f"{self.log_path.stem}_plots.png"
        fig = plt.figure(figsize=(12, 16))
        gs = GridSpec(6, 4, figure=fig)
        epochs = df["epoch"]

        ax1 = fig.add_subplot(gs[0, :])
        _plot_cols(ax1, df, epochs, [("train_loss", "Train"), ("val_loss", "Val")])
        ax1.set_title("Loss")
        ax1.grid(True, alpha=0.3)

        ax2 = fig.add_subplot(gs[1, :], sharex=ax1)
        _plot_cols(ax2, df, epochs, [
            ("direct_precision", "P"), ("direct_recall", "R"), ("direct_f1", "F1"),
        ])
        ax2.set_title("Direct metrics (train forward pass)")
        ax2.grid(True, alpha=0.3)

        ax3 = fig.add_subplot(gs[2, :], sharex=ax1)
        _plot_cols(ax3, df, epochs, [
            ("precision", "P"), ("recall", "R"), ("f1", "F1"),
        ])
        ax3.set_title("CaRB validation metrics")
        ax3.grid(True, alpha=0.3)

        ax4 = fig.add_subplot(gs[3, :], sharex=ax1)
        _plot_cols(ax4, df, epochs, [
            ("train_precision", "P"), ("train_recall", "R"), ("train_f1", "F1"),
        ])
        ax4.set_title("CaRB train subset metrics")
        ax4.grid(True, alpha=0.3)

        for c, name in enumerate(CLASS_NAMES):
            ax = fig.add_subplot(gs[4, c], sharex=ax1)
            _plot_cols(ax, df, epochs, [
                (f"train_precision_{name}", "P"),
                (f"train_recall_{name}", "R"),
                (f"train_f1_{name}", "F1"),
            ])
            ax.set_title(f"Train: {name}")
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.3)

        for c, name in enumerate(CLASS_NAMES):
            ax = fig.add_subplot(gs[5, c], sharex=ax1)
            _plot_cols(ax, df, epochs, [
                (f"precision_{name}", "P"),
                (f"recall_{name}", "R"),
                (f"f1_{name}", "F1"),
            ])
            ax.set_xlabel("Epoch")
            ax.set_title(f"Val: {name}")
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()

    def _plot_per_t_loss(self, per_t_loss: torch.Tensor, epoch: int):
        if self.log_path is None:
            return
        plot_path = self.log_path.parent / "per_t_loss.png"
        t_vals = torch.arange(1, len(per_t_loss) + 1).numpy()
        vals = per_t_loss.cpu().numpy()

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(t_vals, vals, width=0.8)
        ax.set_xlabel("Timestep t")
        ax.set_ylabel("Avg loss")
        ax.set_title(f"Per-timestep loss (epoch {epoch})")
        ax.grid(True, alpha=0.3)
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

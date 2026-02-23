"""Base trainer class for model-agnostic training infrastructure."""
import os
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, List

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pydantic import BaseModel, model_validator
from tqdm import tqdm


class BaseTrainer(ABC):
    """
    Base trainer class that provides common training infrastructure.

    Subclasses should implement:
    - train_step: Model-specific training step
    - validate_step: Model-specific validation step (for inference)
    - validate_loss_step: Model-specific validation step (for loss only)
    - get_trainable_models: Return list of models to optimize
    - get_eval_models: Return list of models to set to eval mode
    - get_checkpoint_state_dict: Return state dict for checkpointing
    - load_checkpoint_state_dict: Load state dict from checkpoint
    """

    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
    ):
        """
        Args:
            device: Training device
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            max_grad_norm: Maximum gradient norm for clipping
        """
        self.device = device
        self.max_grad_norm = max_grad_norm

        # Training state
        self.current_epoch = 0
        self.global_step = 0

        # Optimizer and LR scheduler (set by subclasses)
        self.optimizer = None
        self.lr_scheduler = None

        # Setup optimizer (must be called by subclass after models are initialized)
        self._setup_optimizer(learning_rate, weight_decay)

    def _setup_optimizer(self, learning_rate: float, weight_decay: float):
        """Setup optimizer with trainable models. Called by __init__."""
        trainable_params = self._get_trainable_parameters()
        if trainable_params:
            self.optimizer = torch.optim.AdamW(
                trainable_params,
                lr=learning_rate,
                weight_decay=weight_decay,
            )
        else:
            raise ValueError(
                "No trainable parameters found. Ensure models are initialized before calling _setup_optimizer."
            )

    def _get_trainable_parameters(self) -> List[torch.nn.Parameter]:
        """Get all trainable parameters from models returned by get_trainable_models()."""
        params = []
        for model in self.get_trainable_models():
            params.extend(model.parameters())
        return params

    @abstractmethod
    def get_trainable_models(self) -> List[nn.Module]:
        """
        Return list of models that should be optimized.

        Returns:
            List of nn.Module instances
        """
        pass

    @abstractmethod
    def get_eval_models(self) -> List[nn.Module]:
        """
        Return list of models that should be set to eval mode during validation.

        Returns:
            List of nn.Module instances
        """
        pass

    @abstractmethod
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """
        Perform a single training step.

        Args:
            batch: Dictionary containing batch data

        Returns:
            Dictionary with loss and other metrics
        """
        pass

    @abstractmethod
    def validate_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Perform a single validation step (no gradient computation).
        Should return predictions and labels for metric calculation.

        Args:
            batch: Dictionary containing batch data

        Returns:
            Dictionary with at least:
                - predictions: Predicted outputs
                - labels: True labels
                - attention_mask: (optional) Mask for padding tokens
        """
        pass

    @abstractmethod
    def validate_loss_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """
        Perform a single validation step that only computes loss (no gradient computation).

        Args:
            batch: Dictionary containing batch data

        Returns:
            Dictionary with loss value
        """
        pass

    def set_lr_scheduler(self, scheduler):
        """Set learning rate scheduler."""
        self.lr_scheduler = scheduler

    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
        log_interval: int = 100,
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            dataloader: DataLoader for training data
            epoch: Current epoch number
            log_interval: Logging interval

        Returns:
            Dictionary with average metrics
        """
        self.current_epoch = epoch

        # Set models to training mode
        for model in self.get_trainable_models():
            model.train()
        for model in self.get_eval_models():
            model.train()

        total_metrics = defaultdict(float)
        num_batches = 0

        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {epoch}",
            leave=False,
        )

        for batch_idx, batch in enumerate(progress_bar):
            metrics = self.train_step(batch)

            # Accumulate metrics
            for key, value in metrics.items():
                total_metrics[key] += value
            num_batches += 1

            # Update progress bar
            progress_bar.set_postfix({k: v for k, v in metrics.items()})

            # # Log periodically
            # if (batch_idx + 1) % log_interval == 0:
            #     metric_str = ", ".join([f"{k}={v:.3g}" for k, v in metrics.items()])
            #     tqdm.write(f"Step {self.global_step}: {metric_str}")

        # Average metrics
        avg_metrics = {
            key: total / num_batches if num_batches > 0 else 0.0
            for key, total in total_metrics.items()
        }

        return avg_metrics

    def validate_loss(
        self,
        val_dataloader: DataLoader,
    ) -> Dict[str, float]:
        """
        Compute validation loss over the entire validation dataset.

        Args:
            val_dataloader: DataLoader for validation data

        Returns:
            Dictionary with average validation loss
        """
        # Set models to eval mode
        for model in self.get_eval_models():
            model.eval()

        total_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(
            val_dataloader, desc="Computing validation loss", leave=False
        )

        for batch in progress_bar:
            metrics = self.validate_loss_step(batch)
            total_loss += metrics.get("loss", 0.0)
            num_batches += 1
            progress_bar.set_postfix(metrics)

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        return {
            "val_loss": avg_loss,
        }

    def _compute_component_metrics(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        label_value: int,
    ) -> tuple[int, int, int]:
        """
        Compute overlap and counts for a specific component (subject, object, or predicate).

        Args:
            predictions: Predicted label indices
            labels: True label indices
            label_value: Label value to match (1=subject, 2=object, 3=predicate)

        Returns:
            Tuple of (overlap, predicted_count, gold_count)
        """
        pred_mask = (predictions == label_value).long()
        gold_mask = (labels == label_value).long()
        overlap = (pred_mask * gold_mask).sum().item()
        pred_count = pred_mask.sum().item()
        gold_count = gold_mask.sum().item()
        return overlap, pred_count, gold_count

    def compute_metrics(
        self,
        predictions: torch.Tensor,  # [B, L] or [N]
        labels: torch.Tensor,  # [B, L] or [N]
        attention_mask: Optional[torch.Tensor] = None,  # [B, L] or [N]
    ) -> Dict[str, float]:
        """
        Compute CaRB-style metrics based on token overlap between predicted and gold extractions.

        Args:
            predictions: Predicted label indices [B, L] or [N]
                (0=O, 1=subject, 2=object, 3=predicate)
            labels: True label indices [B, L] or [N]
                (0=O, 1=subject, 2=object, 3=predicate)
            attention_mask: Attention mask (1 for real tokens, 0 for padding) [B, L] or [N]

        Returns:
            Dictionary with precision, recall, and F1 based on token overlap
        """
        # Flatten if needed
        if predictions.dim() > 1:
            predictions = predictions.flatten()
        if labels.dim() > 1:
            labels = labels.flatten()
        if attention_mask is not None and attention_mask.dim() > 1:
            attention_mask = attention_mask.flatten()

        # Mask out padding tokens if mask is provided
        if attention_mask is not None:
            mask = attention_mask.bool()
            pred_flat = predictions[mask]
            label_flat = labels[mask]
        else:
            pred_flat = predictions
            label_flat = labels

        # Convert to CPU for processing
        pred_flat = pred_flat.cpu()
        label_flat = label_flat.cpu()

        # Compute metrics for each component separately
        # Background = 0, Subject = 1, Object = 2, Predicate = 3
        overlap_bg, pred_bg_count, gold_bg_count = self._compute_component_metrics(
            pred_flat, label_flat, 0
        )
        overlap_subj, pred_subj_count, gold_subj_count = self._compute_component_metrics(
            pred_flat, label_flat, 1
        )
        overlap_obj, pred_obj_count, gold_obj_count = self._compute_component_metrics(
            pred_flat, label_flat, 2
        )
        overlap_pred, pred_pred_count, gold_pred_count = self._compute_component_metrics(
            pred_flat, label_flat, 3
        )

        # Sum overlaps and totals across all components
        total_overlap = overlap_subj + overlap_obj + overlap_pred
        total_predicted = pred_subj_count + pred_obj_count + pred_pred_count
        total_gold = gold_subj_count + gold_obj_count + gold_pred_count

        # Per-class (subject, object, predicate) precision, recall, F1
        def _p_r_f1(overlap: int, pred_count: int, gold_count: int) -> tuple[float, float, float]:
            p = overlap / pred_count if pred_count > 0 else 0.0
            r = overlap / gold_count if gold_count > 0 else 0.0
            f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            return p, r, f

        p_bg, r_bg, f_bg = _p_r_f1(overlap_bg, pred_bg_count, gold_bg_count)
        p_subj, r_subj, f_subj = _p_r_f1(overlap_subj, pred_subj_count, gold_subj_count)
        p_obj, r_obj, f_obj = _p_r_f1(overlap_obj, pred_obj_count, gold_obj_count)
        p_pred, r_pred, f_pred = _p_r_f1(overlap_pred, pred_pred_count, gold_pred_count)

        # Compute CaRB-style metrics (micro-averaged)
        # Precision: sum of overlaps / sum of predicted tokens
        precision = total_overlap / total_predicted if total_predicted > 0 else 0.0

        # Recall: sum of overlaps / sum of gold tokens
        recall = total_overlap / total_gold if total_gold > 0 else 0.0

        # F1: harmonic mean of precision and recall
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            # Per-class metrics (background, subject, object, predicate)
            "precision_bg": float(p_bg),
            "recall_bg": float(r_bg),
            "f1_bg": float(f_bg),
            "precision_subj": float(p_subj),
            "recall_subj": float(r_subj),
            "f1_subj": float(f_subj),
            "precision_obj": float(p_obj),
            "recall_obj": float(r_obj),
            "f1_obj": float(f_obj),
            "precision_pred": float(p_pred),
            "recall_pred": float(r_pred),
            "f1_pred": float(f_pred),
        }

    def validate(self, val_dataloader: DataLoader, max_batches: int | None = None) -> Dict[str, float]:
        """
        Run full inference-like validation.

        Args:
            val_dataloader: DataLoader for validation data
            max_batches: Maximum number of batches to validate on. If None, validate on all batches.

        Returns:
            Dictionary with validation metrics
        """
        # Set models to eval mode
        for model in self.get_eval_models():
            model.eval()

        all_predictions = []
        all_labels = []
        all_masks = []

        progress_bar = tqdm(val_dataloader, desc="Validating", leave=False)

        for batch_idx, batch in enumerate(progress_bar):
            if max_batches is not None and batch_idx >= max_batches:
                break
            results = self.validate_step(batch)
            all_predictions.append(results["predictions"].flatten())
            all_labels.append(results["labels"].flatten())
            if "attention_mask" in results:
                all_masks.append(results["attention_mask"].flatten())
            else:
                # Create dummy mask if not provided
                mask = torch.ones_like(results["predictions"].flatten())
                all_masks.append(mask)

        # Concatenate all batches
        predictions = torch.cat(all_predictions, dim=0)
        labels = torch.cat(all_labels, dim=0)
        attention_mask = torch.cat(all_masks, dim=0)

        # Compute metrics
        metrics = self.compute_metrics(predictions, labels, attention_mask)

        return metrics

    @staticmethod
    def plot_training_log(csv_path: Path, plot_path: Optional[Path] = None) -> None:
        """
        Read training CSV log and save a single PNG with five rows:
        1) Train and validation loss vs epoch.
        2) Train metrics (train_precision, train_recall, train_f1) vs epoch.
        3) Validation metrics (precision, recall, f1) vs epoch.
        4) Train per-class precision, recall, F1: background, subject, relation, object (4 columns).
        5) Validation per-class precision, recall, F1: background, subject, relation, object (4 columns).

        Args:
            csv_path: Path to the CSV log (e.g. training_discrete.csv).
            plot_path: Where to save the PNG. If None, uses same dir as csv_path
                with stem '<csv_stem>_plots.png'.
        """
        if not csv_path.exists():
            return
        df = pd.read_csv(csv_path)
        if df.empty or "epoch" not in df.columns:
            return

        if plot_path is None:
            plot_path = csv_path.parent / f"{csv_path.stem}_plots.png"
        plot_path = Path(plot_path)
        plot_path.parent.mkdir(parents=True, exist_ok=True)

        fig = plt.figure(figsize=(12, 14))
        gs = GridSpec(5, 4, figure=fig)
        epochs = df["epoch"].astype(int)

        # Row 1: train and validation loss
        ax1 = fig.add_subplot(gs[0, :])
        if "loss" in df.columns:
            ax1.plot(epochs, df["loss"], label="Train loss", marker="o", markersize=4)
        if "val_loss" in df.columns:
            ax1.plot(
                epochs,
                df["val_loss"],
                label="Validation loss",
                marker="s",
                markersize=4,
            )
        ax1.set_ylabel("Loss")
        ax1.set_title("Train and validation loss")
        if ax1.get_legend_handles_labels()[1]:
            ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Row 2: train metrics (train_precision, train_recall, train_f1)
        ax2 = fig.add_subplot(gs[1, :], sharex=ax1)
        train_metric_cols = ["train_precision", "train_recall", "train_f1"]
        for col in train_metric_cols:
            if col not in df.columns:
                continue
            valid = df[col].notna()
            if not valid.any():
                continue
            label = col.replace("train_", "")
            ax2.plot(
                df.loc[valid, "epoch"],
                df.loc[valid, col],
                label=label,
                marker="o",
                markersize=4,
            )
        ax2.set_ylabel("Score")
        ax2.set_title("Train metrics (per validation epoch)")
        if ax2.get_legend_handles_labels()[1]:
            ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Row 3: validation metrics (precision, recall, f1)
        ax3 = fig.add_subplot(gs[2, :], sharex=ax1)
        val_metric_cols = ["precision", "recall", "f1"]
        for col in val_metric_cols:
            if col not in df.columns:
                continue
            valid = df[col].notna()
            if not valid.any():
                continue
            ax3.plot(
                df.loc[valid, "epoch"],
                df.loc[valid, col],
                label=col,
                marker="o",
                markersize=4,
            )
        ax3.set_ylabel("Score")
        ax3.set_title("Validation metrics (per validation epoch)")
        if ax3.get_legend_handles_labels()[1]:
            ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Row 4: train per-class precision, recall, F1 (4 columns: bg, subj, relation, obj)
        train_per_class = [
            ("train_precision_bg", "train_recall_bg", "train_f1_bg", "Background"),
            ("train_precision_subj", "train_recall_subj", "train_f1_subj", "Subject"),
            ("train_precision_pred", "train_recall_pred", "train_f1_pred", "Relation"),
            ("train_precision_obj", "train_recall_obj", "train_f1_obj", "Object"),
        ]
        for c, (p_col, r_col, f_col, title) in enumerate(train_per_class):
            ax = fig.add_subplot(gs[3, c], sharex=ax1)
            for col, label, marker in [
                (p_col, "Precision", "o"),
                (r_col, "Recall", "s"),
                (f_col, "F1", "^"),
            ]:
                if col in df.columns:
                    valid = df[col].notna()
                    if valid.any():
                        ax.plot(
                            df.loc[valid, "epoch"],
                            df.loc[valid, col],
                            label=label,
                            marker=marker,
                            markersize=4,
                        )
            ax.set_ylabel("Score")
            ax.set_title(f"Train: {title}")
            if ax.get_legend_handles_labels()[1]:
                ax.legend(loc="lower right", fontsize=7)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1.05)

        # Row 5: validation per-class precision, recall, F1 (4 columns)
        val_per_class = [
            ("precision_bg", "recall_bg", "f1_bg", "Background"),
            ("precision_subj", "recall_subj", "f1_subj", "Subject"),
            ("precision_pred", "recall_pred", "f1_pred", "Relation"),
            ("precision_obj", "recall_obj", "f1_obj", "Object"),
        ]
        for c, (p_col, r_col, f_col, title) in enumerate(val_per_class):
            ax = fig.add_subplot(gs[4, c], sharex=ax1)
            for col, label, marker in [
                (p_col, "Precision", "o"),
                (r_col, "Recall", "s"),
                (f_col, "F1", "^"),
            ]:
                if col in df.columns:
                    valid = df[col].notna()
                    if valid.any():
                        ax.plot(
                            df.loc[valid, "epoch"],
                            df.loc[valid, col],
                            label=label,
                            marker=marker,
                            markersize=4,
                        )
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Score")
            ax.set_title(f"Val: {title}")
            if ax.get_legend_handles_labels()[1]:
                ax.legend(loc="lower right", fontsize=7)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1.05)

        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.setp(ax2.get_xticklabels(), visible=False)
        plt.setp(ax3.get_xticklabels(), visible=False)
        for ax in fig.get_axes()[3:7]:
            plt.setp(ax.get_xticklabels(), visible=False)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()

    def train(
        self,
        train_dataloader: DataLoader,
        num_epochs: int = 11,
        log_interval: int = 100,
        save_path: Optional[str] = None,
        save_interval: int = 1,
        val_dataloader: Optional[DataLoader] = None,
        val_full_interval: int = 5,
        val_metrics_on_train: bool = False,
        log_path: Optional[str] = None,
        train_val_batches: int | None = None,
    ):
        """
        Main training loop.

        Args:
            train_dataloader: DataLoader for training data
            num_epochs: Number of training epochs
            log_interval: Logging interval
            save_path: Path to save checkpoints (optional)
            save_interval: Save checkpoint every N epochs
            val_dataloader: DataLoader for validation data (optional)
            val_full_interval: Run full validation (with metrics) every N epochs.
                Validation loss is computed after every epoch.
            val_metrics_on_train: If True, compute validation-style metrics
                (precision, recall, f1) on the train dataloader at the same
                cadence as full validation. Logged as train_precision,
                train_recall, train_f1.
            log_path: Path to CSV log file (train/val loss and validation metrics).
                If None and save_path is set, uses save_path/train_log.csv.
            train_val_batches: Maximum number of batches to validate on the train dataloader. If None, validate on all batches.
        """
        # Resolve CSV log path (pathlib)
        log_path_resolved = None
        if log_path is not None:
            log_path_resolved = Path(log_path)
        elif save_path is not None:
            log_path_resolved = Path(save_path) / "train_log.csv"

        log_rows = []

        # Get model info for logging (trainable params only; frozen e.g. BERT excluded)
        total_params = sum(
            p.numel()
            for model in self.get_trainable_models()
            for p in model.parameters()
            if p.requires_grad
        )

        print(f"Starting training for {num_epochs} epochs on {self.device}")
        print(f"Trainable parameters: {total_params:,}")
        if val_dataloader is not None:
            print(f"Validation enabled with {len(val_dataloader.dataset)} examples")
            print(
                f"Validation loss computed every epoch, full validation every {val_full_interval} epochs"
            )
        if val_metrics_on_train:
            print(
                f"Train metrics (precision/recall/f1) computed every {val_full_interval} epochs"
            )
        if log_path_resolved is not None:
            print(f"CSV log: {log_path_resolved}")

        # Track best validation F1 for saving best model
        best_f1 = -1.0

        for epoch in range(1, num_epochs + 1):
            # Training
            train_metrics = self.train_epoch(train_dataloader, epoch, log_interval)

            # Validation loss (computed after every epoch)
            val_loss_metrics = None
            if val_dataloader is not None:
                val_loss_metrics = self.validate_loss(val_dataloader)

            # Full validation with metrics (computed every N epochs)
            val_metrics = None
            if val_dataloader is not None and epoch % val_full_interval == 0:
                val_metrics = self.validate(val_dataloader)

            # Validation-style metrics on train dataloader (optional, same cadence)
            train_metrics_full = None
            if val_metrics_on_train and epoch % val_full_interval == 0:
                train_metrics_full = self.validate(train_dataloader, max_batches=train_val_batches)

            # CSV log: append row from train_metrics, val_loss_metrics, val_metrics
            if log_path_resolved is not None:
                row = {"epoch": epoch}
                train_metrics = {f"train_direct_{k}":v for k, v in train_metrics.items()}
                row.update(train_metrics)
                if val_loss_metrics is not None:
                    row.update(val_loss_metrics)
                if val_metrics is not None:
                    row.update(val_metrics)
                if train_metrics_full is not None:
                    train_metrics_full = {f"train_{k}":v for k, v in train_metrics_full.items()}
                    row.update(
                        train_metrics_full
                    )
                log_rows.append(row)
                df = pd.DataFrame(log_rows)
                log_path_resolved.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(log_path_resolved, index=False)
                self.plot_training_log(log_path_resolved)


            # Print metrics with colors; a big mess maybe better to do in sep func
            _c = {"r": "\033[0m", "b": "\033[1m", "dim": "\033[2m", "cyan": "\033[36m", "green": "\033[32m", "yellow": "\033[33m"}
            _fmt = lambda v: f"{v:.4g}" if isinstance(v, (int, float)) else str(v)

            rows = []
            for k, v in train_metrics.items():
                rows.append((f"train {k}", v))
            if train_metrics_full is not None:
                for key in ("train_precision", "train_recall", "train_f1"):
                    if key in train_metrics_full:
                        rows.append((key, train_metrics_full[key]))
            if val_loss_metrics is not None:
                for k, v in val_loss_metrics.items():
                    rows.append((f"val {k}", v))
            if val_metrics is not None:
                for key in ("precision", "recall", "f1"):
                    if key in val_metrics:
                        rows.append((f"val_full {key}", val_metrics[key]))

            label_w = max(len(label) for label, _ in rows) if rows else 0
            prefix = "  "

            print(f"\n{_c['b']}{_c['cyan']}Epoch {epoch}/{num_epochs}{_c['r']}")
            for label, value in rows:
                pad = " " * (label_w - len(label))
                print(f"{prefix}{_c['dim']}{label}{_c['r']}{pad}\t{_fmt(value)}")

            if val_metrics is not None:
                if save_path and val_metrics['f1'] > best_f1:
                    best_f1 = val_metrics['f1']
                    self.save_checkpoint(
                        save_path, epoch, suffix="best", extra_info={"best_f1": best_f1}
                    )
                    print(f"  {_c['green']}New best F1: {best_f1:.4f}, saved checkpoint{_c['r']}")

            # Save checkpoint
            if save_path and epoch % save_interval == 0:
                self.save_checkpoint(save_path, epoch)


    @abstractmethod
    def get_checkpoint_state_dict(self) -> Dict[str, torch.Tensor]:
        """
        Get state dictionaries for checkpointing.

        Returns:
            Dictionary mapping keys to state dicts
        """
        pass

    @abstractmethod
    def load_checkpoint_state_dict(self, checkpoint: Dict[str, torch.Tensor]):
        """
        Load state dictionaries from checkpoint.

        Args:
            checkpoint: Dictionary mapping keys to state dicts
        """
        pass

    def save_checkpoint(
        self,
        path: str,
        epoch: int,
        suffix: Optional[str] = None,
        extra_info: Optional[Dict] = None,
    ):
        """
        Save training checkpoint.

        Args:
            path: Directory path to save checkpoint
            epoch: Current epoch number
            suffix: Optional suffix for checkpoint filename (e.g., "best")
            extra_info: Optional dictionary with additional info to save in checkpoint
        """
        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "optimizer_state_dict": self.optimizer.state_dict()
            if self.optimizer is not None
            else None,
        }
        checkpoint.update(self.get_checkpoint_state_dict())
        if extra_info:
            checkpoint.update(extra_info)
        os.makedirs(path, exist_ok=True)
        if suffix:
            filename = f"{path}/checkpoint_{suffix}.pt"
        else:
            filename = f"{path}/checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, filename)
        print(f"Checkpoint saved to {filename}")

    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.load_checkpoint_state_dict(checkpoint)
        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        if (
            self.optimizer is not None
            and "optimizer_state_dict" in checkpoint
            and checkpoint["optimizer_state_dict"] is not None
        ):
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"Checkpoint loaded from {path}")


class BaseTrainerConfig(BaseModel):
    """
    Configuration model for BaseTrainer.
    Acts as a factory for creating BaseTrainer instances.
    """

    device: Optional[str] = None  # None = auto-detect (cuda if available, else cpu)
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    @model_validator(mode="after")
    def _auto_detect_device(self):
        """Auto-detect device if not specified."""
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        return self

    def create_trainer(self, model: nn.Module) -> BaseTrainer:
        raise NotImplementedError("Subclasses must implement create_trainer method")
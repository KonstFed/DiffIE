"""Base trainer class for model-agnostic training infrastructure."""
import os
from typing import Dict, Optional, List
from abc import ABC, abstractmethod
from collections import defaultdict

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

            # Log periodically
            if (batch_idx + 1) % log_interval == 0:
                metric_str = ", ".join([f"{k}={v:.3g}" for k, v in metrics.items()])
                tqdm.write(f"Step {self.global_step}: {metric_str}")

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
        num_classes: int = 4,
    ) -> Dict[str, float]:
        """
        Compute CaRB-style metrics based on token overlap between predicted and gold extractions.

        Args:
            predictions: Predicted label indices [B, L] or [N]
                (0=O, 1=subject, 2=object, 3=predicate)
            labels: True label indices [B, L] or [N]
                (0=O, 1=subject, 2=object, 3=predicate)
            attention_mask: Attention mask (1 for real tokens, 0 for padding) [B, L] or [N]
            num_classes: Number of label classes (unused, kept for compatibility)

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
        # Subject = 1, Object = 2, Predicate = 3
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
        }

    def validate(
        self,
        val_dataloader: DataLoader,
        num_classes: int = 4,
    ) -> Dict[str, float]:
        """
        Run full inference-like validation.

        Args:
            val_dataloader: DataLoader for validation data
            num_classes: Number of label classes for validation metrics

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

        for batch in progress_bar:
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
        metrics = self.compute_metrics(predictions, labels, attention_mask, num_classes)

        return metrics

    def train(
        self,
        train_dataloader: DataLoader,
        num_epochs: int = 11,
        log_interval: int = 100,
        save_path: Optional[str] = None,
        save_interval: int = 1,
        val_dataloader: Optional[DataLoader] = None,
        num_classes: int = 4,
        val_full_interval: int = 5,
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
            num_classes: Number of label classes for validation metrics
            val_full_interval: Run full validation (with metrics) every N epochs.
                Validation loss is computed after every epoch.
        """
        # Get model info for logging
        total_params = sum(
            p.numel()
            for model in self.get_trainable_models()
            for p in model.parameters()
        )

        print(f"Starting training for {num_epochs} epochs on {self.device}")
        print(f"Model parameters: {total_params:,}")
        if val_dataloader is not None:
            print(f"Validation enabled with {len(val_dataloader.dataset)} examples")
            print(
                f"Validation loss computed every epoch, full validation every {val_full_interval} epochs"
            )

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
                val_metrics = self.validate(val_dataloader, num_classes)

            # Print metrics with smart formatting (up to 20 significant digits, auto scientific)
            metric_str = ", ".join([f"{k}={v:.20g}" for k, v in train_metrics.items()])
            print(f"Epoch {epoch}/{num_epochs} completed. Train: {metric_str}")

            if val_loss_metrics is not None:
                print(f"  Val loss: {val_loss_metrics['val_loss']:.20g}")

            if val_metrics is not None:
                print(
                    f"  Val Precision: {val_metrics['precision']:.20g}, "
                    f"Val Recall: {val_metrics['recall']:.20g}, "
                    f"Val F1: {val_metrics['f1']:.20g}"
                )

                # Save best model based on F1 score
                if save_path and val_metrics['f1'] > best_f1:
                    best_f1 = val_metrics['f1']
                    self.save_checkpoint(
                        save_path, epoch, suffix="best", extra_info={"best_f1": best_f1}
                    )
                    print(f"  New best F1: {best_f1:.4f}, saved best checkpoint")

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

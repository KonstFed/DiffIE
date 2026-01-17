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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


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
                metric_str = ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
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

    def compute_metrics(
        self,
        predictions: torch.Tensor,  # [B, L] or [N]
        labels: torch.Tensor,  # [B, L] or [N]
        attention_mask: Optional[torch.Tensor] = None,  # [B, L] or [N]
        num_classes: int = 4,
    ) -> Dict[str, float]:
        """
        Compute sequence labeling metrics.

        Args:
            predictions: Predicted label indices [B, L] or [N]
            labels: True label indices [B, L] or [N]
            attention_mask: Attention mask (1 for real tokens, 0 for padding) [B, L] or [N]
            num_classes: Number of label classes

        Returns:
            Dictionary with accuracy, precision, recall, F1 (per class and macro-averaged)
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
            pred_flat = predictions[mask].cpu().numpy()
            label_flat = labels[mask].cpu().numpy()
        else:
            pred_flat = predictions.cpu().numpy()
            label_flat = labels.cpu().numpy()

        # Convert to numpy arrays for sklearn
        # Ensure all class indices are present (sklearn needs this for proper averaging)
        labels_list = list(range(num_classes))

        # Compute accuracy
        accuracy = accuracy_score(label_flat, pred_flat)

        # Compute per-class and macro-averaged metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            label_flat,
            pred_flat,
            labels=labels_list,
            average=None,  # Returns per-class metrics
            zero_division=0.0,
        )

        # Compute macro-averaged metrics
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            label_flat,
            pred_flat,
            labels=labels_list,
            average="macro",
            zero_division=0.0,
        )

        # Compute weighted F1 (weighted by class frequency)
        _, _, weighted_f1, _ = precision_recall_fscore_support(
            label_flat,
            pred_flat,
            labels=labels_list,
            average="weighted",
            zero_division=0.0,
        )

        # Convert to dictionaries for per-class metrics
        class_precision = {i: float(precision[i]) for i in range(num_classes)}
        class_recall = {i: float(recall[i]) for i in range(num_classes)}
        class_f1 = {i: float(f1[i]) for i in range(num_classes)}

        return {
            "accuracy": float(accuracy),
            "macro_precision": float(macro_precision),
            "macro_recall": float(macro_recall),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
            "class_precision": class_precision,
            "class_recall": class_recall,
            "class_f1": class_f1,
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

            # Print metrics
            metric_str = ", ".join([f"{k}={v:.4f}" for k, v in train_metrics.items()])
            print(f"Epoch {epoch}/{num_epochs} completed. Train: {metric_str}")

            if val_loss_metrics is not None:
                print(f"  Val loss: {val_loss_metrics['val_loss']:.4f}")

            if val_metrics is not None:
                print(
                    f"  Val accuracy: {val_metrics['accuracy']:.4f}, "
                    f"Val macro F1: {val_metrics['macro_f1']:.4f}, "
                    f"Val weighted F1: {val_metrics['weighted_f1']:.4f}"
                )
                # Print per-class F1 scores
                print("  Per-class F1:", end=" ")
                for class_idx in range(num_classes):
                    class_name = (
                        ["O", "Subject", "Object", "Predicate"][class_idx]
                        if class_idx < 4
                        else f"Class{class_idx}"
                    )
                    print(
                        f"{class_name}: {val_metrics['class_f1'][class_idx]:.4f}",
                        end="  ",
                    )
                print()

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

    def save_checkpoint(self, path: str, epoch: int):
        """Save training checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "optimizer_state_dict": self.optimizer.state_dict()
            if self.optimizer is not None
            else None,
        }
        checkpoint.update(self.get_checkpoint_state_dict())
        os.makedirs(path, exist_ok=True)
        torch.save(checkpoint, f"{path}/checkpoint_epoch_{epoch}.pt")
        print(f"Checkpoint saved to {path}/checkpoint_epoch_{epoch}.pt")

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

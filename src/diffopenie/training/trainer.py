"""Training loop for diffusion-based OpenIE model."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from typing import Dict, Optional
from collections import defaultdict

from tqdm import tqdm

from diffopenie.models.diffusion.denoiser import DiffusionSLDenoiser
from diffopenie.models.diffusion.scheduler import LinearScheduler
from diffopenie.models.label_mapper import LabelMapper
from diffopenie.models.encoder import BERTEncoder


class DiffusionTrainer:
    """
    Trainer for diffusion-based sequence labeling model.
    
    Handles:
    - Forward diffusion process (adding noise)
    - Denoising model training
    - Loss computation (MSE on x0 prediction)
    """
    def __init__(
        self,
        denoiser: DiffusionSLDenoiser,
        scheduler: LinearScheduler,
        label_mapper: LabelMapper,
        encoder: BERTEncoder,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
    ):
        """
        Args:
            denoiser: Diffusion denoiser model
            scheduler: Diffusion scheduler (handles noise schedule)
            label_mapper: Maps label indices to embeddings
            encoder: BERT encoder for token embeddings
            device: Training device
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
        """
        self.denoiser = denoiser.to(device)
        self.scheduler = scheduler.to(device)  # Scheduler is now nn.Module, so .to(device) moves all buffers
        self.label_mapper = label_mapper.to(device)
        self.encoder = encoder.to(device)
        self.device = device
        
        # Setup optimizer
        self.optimizer = AdamW(
            self.denoiser.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        
        # Optional learning rate scheduler (can be set via set_lr_scheduler)
        self.lr_scheduler = None
        
        # Loss function
        self.criterion = nn.MSELoss(reduction="mean")
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
    
    def set_lr_scheduler(self, scheduler):
        """Set learning rate scheduler."""
        self.lr_scheduler = scheduler
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """
        Perform a single training step.
        
        Args:
            batch: Dictionary containing:
                - token_ids: [B, L] token IDs
                - attention_mask: [B, L] attention mask
                - label_indices: [B, L] label indices
        
        Returns:
            Dictionary with loss and other metrics
        """
        self.denoiser.train()
        self.encoder.train()
        
        # Move batch to device
        token_ids = batch["token_ids"].to(self.device)  # [B, L]
        attention_mask = batch["attention_mask"].to(self.device)  # [B, L]
        label_indices = batch["label_indices"].to(self.device)  # [B, L]
        
        B, L = token_ids.shape
        
        # Get BERT token embeddings
        token_embeddings = self.encoder(token_ids, attention_mask)  # [B, L, bert_dim]
        
        # Convert label indices to embeddings (x_0)
        x_0 = self.label_mapper(label_indices)  # [B, L, x_dim]
        
        # Sample random timesteps
        t = torch.randint(
            0,
            self.scheduler.num_steps,
            size=(B,),
            device=self.device,
            dtype=torch.long,
        )  # [B]
        
        # Sample noise
        noise = torch.randn_like(x_0)  # [B, L, x_dim]
        
        # Forward diffusion: add noise to x_0
        x_t = self.scheduler.q_sample(x_0, t, noise)  # [B, L, x_dim]
        
        # Predict x_0 from x_t
        x0_pred = self.denoiser(
            x_t=x_t,
            t=t,
            token_embeddings=token_embeddings,
            attn_mask=attention_mask.bool(),
        )  # [B, L, x_dim]
        
        # Compute loss: MSE between predicted and true x_0
        # Only compute loss on non-padding tokens
        mask = attention_mask.unsqueeze(-1).expand_as(x_0)  # [B, L, x_dim]
        loss = self.criterion(x0_pred * mask, x_0 * mask)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.denoiser.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Update learning rate scheduler if available
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        
        self.global_step += 1
        
        return {
            "loss": loss.item(),
        }
    
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
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {epoch}",
            leave=False,
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            metrics = self.train_step(batch)
            total_loss += metrics["loss"]
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({"loss": metrics["loss"]})
            
            # Log periodically
            if (batch_idx + 1) % log_interval == 0:
                tqdm.write(
                    f"Step {self.global_step}: loss={metrics['loss']:.4f}"
                )
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        return {
            "loss": avg_loss,
        }
    
    def validate_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Perform a single validation step (no gradient computation).
        Uses the denoiser's inference method for step-by-step denoising.
        
        Args:
            batch: Dictionary containing:
                - token_ids: [B, L] token IDs
                - attention_mask: [B, L] attention mask
                - label_indices: [B, L] label indices
        
        Returns:
            Dictionary with predictions and labels for metric calculation
        """
        self.scheduler.eval()
        self.denoiser.eval()
        self.encoder.eval()

        
        with torch.no_grad():
            # Move batch to device
            token_ids = batch["token_ids"].to(self.device)  # [B, L]
            attention_mask = batch["attention_mask"].to(self.device)  # [B, L]
            label_indices = batch["label_indices"].to(self.device)  # [B, L]
            
            # Get BERT token embeddings
            token_embeddings = self.encoder(token_ids, attention_mask)  # [B, L, bert_dim]
            
            # TODO: idea: make fixed noise for validation

            # Perform step-by-step inference (same as inference)
            noise_shape = (token_ids.shape[0], token_ids.shape[1], self.label_mapper.embedding_dim)
            x0_pred = self.scheduler.inference(
                denoiser=self.denoiser,
                shape=noise_shape,
                condition=token_embeddings,
            )  # [B, L, x_dim]
            
            # Convert predictions back to label indices
            pred_indices = self.label_mapper.reverse(x0_pred)  # [B, L]
            
            return {
                "predictions": pred_indices,
                "labels": label_indices,
                "attention_mask": attention_mask,
            }
    
    def validate_loss_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """
        Perform a single validation step that only computes loss (no gradient computation).
        Similar to train_step but without backward pass.
        
        Args:
            batch: Dictionary containing:
                - token_ids: [B, L] token IDs
                - attention_mask: [B, L] attention mask
                - label_indices: [B, L] label indices
        
        Returns:
            Dictionary with loss value
        """
        self.denoiser.eval()
        self.encoder.eval()
        
        with torch.no_grad():
            # Move batch to device
            token_ids = batch["token_ids"].to(self.device)  # [B, L]
            attention_mask = batch["attention_mask"].to(self.device)  # [B, L]
            label_indices = batch["label_indices"].to(self.device)  # [B, L]
            
            B, L = token_ids.shape
            
            # Get BERT token embeddings
            token_embeddings = self.encoder(token_ids, attention_mask)  # [B, L, bert_dim]
            
            # Convert label indices to embeddings (x_0)
            x_0 = self.label_mapper(label_indices)  # [B, L, x_dim]
            
            # Sample random timesteps
            t = torch.randint(
                0,
                self.scheduler.num_steps,
                size=(B,),
                device=self.device,
                dtype=torch.long,
            )  # [B]
            
            # Sample noise
            noise = torch.randn_like(x_0)  # [B, L, x_dim]
            
            # Forward diffusion: add noise to x_0
            x_t = self.scheduler.q_sample(x_0, t, noise)  # [B, L, x_dim]
            
            # Predict x_0 from x_t
            x0_pred = self.denoiser(
                x_t=x_t,
                t=t,
                token_embeddings=token_embeddings,
                attn_mask=attention_mask.bool(),
            )  # [B, L, x_dim]
            
            # Compute loss: MSE between predicted and true x_0
            # Only compute loss on non-padding tokens
            mask = attention_mask.unsqueeze(-1).expand_as(x_0)  # [B, L, x_dim]
            loss = self.criterion(x0_pred * mask, x_0 * mask)
            
            return {
                "loss": loss.item(),
            }
    
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
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(val_dataloader, desc="Computing validation loss", leave=False)
        
        for batch in progress_bar:
            metrics = self.validate_loss_step(batch)
            total_loss += metrics["loss"]
            num_batches += 1
            progress_bar.set_postfix({"loss": metrics["loss"]})
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        return {
            "val_loss": avg_loss,
        }
    
    def compute_metrics(
        self,
        predictions: torch.Tensor,  # [B, L]
        labels: torch.Tensor,  # [B, L]
        attention_mask: torch.Tensor,  # [B, L]
        num_classes: int = 4,
    ) -> Dict[str, float]:
        """
        Compute sequence labeling metrics.
        
        Args:
            predictions: Predicted label indices [B, L]
            labels: True label indices [B, L]
            attention_mask: Attention mask (1 for real tokens, 0 for padding) [B, L]
            num_classes: Number of label classes
        
        Returns:
            Dictionary with accuracy, precision, recall, F1 (per class and macro-averaged)
        """
        # Flatten and mask out padding tokens
        mask = attention_mask.bool()
        pred_flat = predictions[mask].cpu()
        label_flat = labels[mask].cpu()
        
        # Overall accuracy
        correct = (pred_flat == label_flat).sum().item()
        total = pred_flat.numel()
        accuracy = correct / total if total > 0 else 0.0
        
        # Per-class metrics
        class_metrics = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
        
        for class_idx in range(num_classes):
            pred_class = (pred_flat == class_idx)
            label_class = (label_flat == class_idx)
            
            tp = (pred_class & label_class).sum().item()
            fp = (pred_class & ~label_class).sum().item()
            fn = (~pred_class & label_class).sum().item()
            
            class_metrics[class_idx] = {
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
        
        # Calculate precision, recall, F1 per class
        class_precision = {}
        class_recall = {}
        class_f1 = {}
        
        for class_idx in range(num_classes):
            metrics = class_metrics[class_idx]
            tp, fp, fn = metrics["tp"], metrics["fp"], metrics["fn"]
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            class_precision[class_idx] = precision
            class_recall[class_idx] = recall
            class_f1[class_idx] = f1
        
        # Macro-averaged metrics
        macro_precision = sum(class_precision.values()) / num_classes
        macro_recall = sum(class_recall.values()) / num_classes
        macro_f1 = sum(class_f1.values()) / num_classes
        
        # Weighted F1 (weighted by class frequency)
        class_counts = torch.bincount(label_flat, minlength=num_classes).float()
        class_weights = class_counts / class_counts.sum() if class_counts.sum() > 0 else torch.ones(num_classes) / num_classes
        weighted_f1 = sum(class_f1[i] * class_weights[i].item() for i in range(num_classes))
        
        return {
            "accuracy": accuracy,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
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
        Run full inference like validation.
        
        Args:
            val_dataloader: DataLoader for validation data
            num_classes: Number of label classes
        
        Returns:
            Dictionary with validation metrics
        """
        all_predictions = []
        all_labels = []
        all_masks = []
        
        progress_bar = tqdm(val_dataloader, desc="Validating", leave=False)
        
        for batch in progress_bar:
            results = self.validate_step(batch)
            # flatten and make "one sequence"
            all_predictions.append(results["predictions"].flatten())
            all_labels.append(results["labels"].flatten())
            all_masks.append(results["attention_mask"].flatten())
        
        # Concatenate all batches
        predictions = torch.cat(all_predictions, dim=0)  # [N, L]
        labels = torch.cat(all_labels, dim=0)  # [N, L]
        attention_mask = torch.cat(all_masks, dim=0)  # [N, L]
        
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
        print(f"Starting training for {num_epochs} epochs on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.denoiser.parameters()):,}")
        if val_dataloader is not None:
            print(f"Validation enabled with {len(val_dataloader.dataset)} examples")
            print(f"Validation loss computed every epoch, full validation every {val_full_interval} epochs")
        
        for epoch in range(1, num_epochs + 1):
            # Training
            # train_metrics = self.train_epoch(train_dataloader, epoch, log_interval)
            train_metrics = {"loss": 0.0}
            
            # Validation loss (computed after every epoch)
            val_loss_metrics = None
            if val_dataloader is not None:
                val_loss_metrics = self.validate_loss(val_dataloader)
            
            # Full validation with metrics (computed every N epochs)
            val_metrics = None
            if val_dataloader is not None and epoch % val_full_interval == 0:
                val_metrics = self.validate(val_dataloader, num_classes)
            
            # Print metrics
            print(
                f"Epoch {epoch}/{num_epochs} completed. "
                f"Train loss: {train_metrics['loss']:.4f}"
            )
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
                    class_name = ["O", "Subject", "Object", "Predicate"][class_idx] if class_idx < 4 else f"Class{class_idx}"
                    print(f"{class_name}: {val_metrics['class_f1'][class_idx]:.4f}", end="  ")
                print()
            
            # Save checkpoint
            if save_path and epoch % save_interval == 0:
                self.save_checkpoint(save_path, epoch)
    
    def save_checkpoint(self, path: str, epoch: int):
        """Save training checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "denoiser_state_dict": self.denoiser.state_dict(),
            "label_mapper_state_dict": self.label_mapper.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        torch.save(checkpoint, f"{path}/checkpoint_epoch_{epoch}.pt")
        print(f"Checkpoint saved to {path}/checkpoint_epoch_{epoch}.pt")
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.denoiser.load_state_dict(checkpoint["denoiser_state_dict"])
        self.label_mapper.load_state_dict(checkpoint["label_mapper_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        print(f"Checkpoint loaded from {path}")

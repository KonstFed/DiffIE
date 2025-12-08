"""Training loop for diffusion-based OpenIE model."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from typing import Dict, Optional

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
        self.scheduler = scheduler
        self.label_mapper = label_mapper.to(device)
        self.encoder = encoder.to(device)
        self.device = device
        
        # Move scheduler tensors to device
        self._move_scheduler_to_device()
        
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
    
    def _move_scheduler_to_device(self):
        """Move scheduler tensors to the correct device."""
        for attr in [
            "betas", "alphas", "alphas_cumprod", "sqrt_alpha_cumprod",
            "sqrt_one_minus_alpha_cumprod", "alphas_cumprod_prev", "posterior_variance"
        ]:
            if hasattr(self.scheduler, attr):
                tensor = getattr(self.scheduler, attr)
                if isinstance(tensor, torch.Tensor):
                    setattr(self.scheduler, attr, tensor.to(self.device))
    
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
    
    def train(
        self,
        train_dataloader: DataLoader,
        num_epochs: int = 10,
        log_interval: int = 100,
        save_path: Optional[str] = None,
        save_interval: int = 1,
    ):
        """
        Main training loop.
        
        Args:
            train_dataloader: DataLoader for training data
            num_epochs: Number of training epochs
            log_interval: Logging interval
            save_path: Path to save checkpoints (optional)
            save_interval: Save checkpoint every N epochs
        """
        print(f"Starting training for {num_epochs} epochs on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.denoiser.parameters()):,}")
        
        for epoch in range(1, num_epochs + 1):
            metrics = self.train_epoch(train_dataloader, epoch, log_interval)
            
            print(
                f"Epoch {epoch}/{num_epochs} completed. "
                f"Average loss: {metrics['loss']:.4f}"
            )
            
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

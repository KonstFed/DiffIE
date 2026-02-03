"""Training loop for diffusion-based OpenIE model."""
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from diffopenie.models.span import SpanDiffusionModel


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
        model: SpanDiffusionModel,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
    ):
        """
        Args:
            model: Unified diffusion sequence labeler model
            device: Training device
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            max_grad_norm: Maximum gradient norm for clipping
        """
        # Move model to device
        self.model = model.to(device)

        # Initialize base trainer (sets up optimizer)
        super().__init__(
            device=device,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
        )

        # Loss function
        self.criterion = nn.MSELoss(reduction="none")

    def get_trainable_models(self) -> List[nn.Module]:
        """Return list of models that should be optimized."""
        return [self.model.denoiser]

    def get_eval_models(self) -> List[nn.Module]:
        """Return list of models that should be set to eval mode during validation."""
        return [self.model.scheduler, self.model.encoder]

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
                - label_spans: [B, 6] label indices

        Returns:
            Dictionary with loss and other metrics
        """
        # Move batch to device
        token_ids = batch["token_ids"].to(self.device)  # [B, L]
        attention_mask = batch["attention_mask"].to(self.device)  # [B, L]
        label_spans = batch["label_spans"].to(self.device)  # [B, 6]

        B, L = token_ids.shape

        # Get BERT token embeddings
        token_embeddings = self.model.encode_tokens(
            token_ids, attention_mask
        )  # [B, L, bert_dim]
        # Sample random timesteps
        t = torch.randint(
            0,
            self.model.scheduler.num_steps,
            size=(B,),
            device=self.device,
            dtype=torch.long,
        )  # [B]

        # Forward diffusion: add noise to x_0
        x_0 = self.model.label_mapper.forward(label_spans) # [B, 6, L]
        x_t = self.model.noise(label_spans, t)  # [B, 6, L]

        # Predict x_0 from x_t
        x0_pred = self.model.denoiser(
            x_t=x_t,
            t=t,
            token_embeddings=token_embeddings,
            attn_mask=attention_mask.bool(),
        )  # [B, 6, L]

        # Compute loss: MSE between predicted and true x_0
        # Only compute loss on non-padding tokens
        # TODO: if doesn't work manually debug it or comment attention mask aware loss.
        mask = attention_mask.unsqueeze(1).expand_as(x_0)  # [B, 6, L]
        loss_per_element = self.criterion(x0_pred, x_0)  # [B, 6, L]
        weighted_loss = loss_per_element * mask
        loss = weighted_loss.sum() / mask.sum().clamp(min=1)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping - clip all trainable parameters
        # trainable_params = []
        # for model in self.get_trainable_models():
        #     trainable_params.extend(model.parameters())
        # torch.nn.utils.clip_grad_norm_(
        #     trainable_params, max_norm=self.max_grad_norm
        # )

        self.optimizer.step()

        # Update learning rate scheduler if available
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        self.global_step += 1

        return {
            "loss": loss.item(),
        }


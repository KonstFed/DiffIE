"""Training loop for diffusion-based OpenIE model."""
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from diffopenie.models.diffusion_model import DiffusionSequenceLabeler
from diffopenie.training.base_trainer import BaseTrainer, BaseTrainerConfig


class DiffusionTrainer(BaseTrainer):
    """
    Trainer for diffusion-based sequence labeling model.

    Handles:
    - Forward diffusion process (adding noise)
    - Denoising model training
    - Loss computation (MSE on x0 prediction)
    """

    def __init__(
        self,
        model: DiffusionSequenceLabeler,
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
        return [self.model.denoiser, self.model.label_mapper]

    def get_eval_models(self) -> List[nn.Module]:
        """Return list of models that should be set to eval mode during validation."""
        return [self.model.scheduler, self.model.denoiser, self.model.encoder]

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
        # Move batch to device
        token_ids = batch["token_ids"].to(self.device)  # [B, L]
        attention_mask = batch["attention_mask"].to(self.device)  # [B, L]
        label_indices = batch["label_indices"].to(self.device)  # [B, L]

        B, L = token_ids.shape

        # Get BERT token embeddings
        token_embeddings = self.model.encode_tokens(
            token_ids, attention_mask
        )  # [B, L, bert_dim]

        # Convert label indices to embeddings (x_0)
        x_0 = self.model.labels_to_embeddings(label_indices)  # [B, L, x_dim]

        # Sample random timesteps
        t = torch.randint(
            0,
            self.model.scheduler.num_steps,
            size=(B,),
            device=self.device,
            dtype=torch.long,
        )  # [B]

        # Sample noise
        noise = torch.randn_like(x_0)  # [B, L, x_dim]

        # Forward diffusion: add noise to x_0
        x_t = self.model.scheduler.q_sample(x_0, t, noise)  # [B, L, x_dim]

        # Predict x_0 from x_t
        x0_pred = self.model.denoiser(
            x_t=x_t,
            t=t,
            token_embeddings=token_embeddings,
            attn_mask=attention_mask.bool(),
        )  # [B, L, x_dim]

        # Compute loss: MSE between predicted and true x_0
        # Only compute loss on non-padding tokens
        mask = attention_mask.unsqueeze(-1).expand_as(x_0)  # [B, L, x_dim]
        loss_per_element = self.criterion(x0_pred, x_0)  # [B, L, x_dim]
        loss = (loss_per_element * mask).sum() / mask.sum()  # Mean over valid tokens

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping - clip all trainable parameters
        trainable_params = []
        for model in self.get_trainable_models():
            trainable_params.extend(model.parameters())
        torch.nn.utils.clip_grad_norm_(
            trainable_params, max_norm=self.max_grad_norm
        )

        self.optimizer.step()

        # Update learning rate scheduler if available
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        self.global_step += 1

        return {
            "loss": loss.item(),
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
        with torch.no_grad():
            # Move batch to device
            token_ids = batch["token_ids"].to(self.device)  # [B, L]
            attention_mask = batch["attention_mask"].to(self.device)  # [B, L]
            label_indices = batch["label_indices"].to(self.device)  # [B, L]

            # TODO: idea: make fixed noise for validation

            # Perform inference using the model's predict method
            pred_indices = self.model.predict(token_ids, attention_mask)  # [B, L]

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
        with torch.no_grad():
            # Move batch to device
            token_ids = batch["token_ids"].to(self.device)  # [B, L]
            attention_mask = batch["attention_mask"].to(self.device)  # [B, L]
            label_indices = batch["label_indices"].to(self.device)  # [B, L]

            B, L = token_ids.shape

            # Get BERT token embeddings
            token_embeddings = self.model.encode_tokens(
                token_ids, attention_mask
            )  # [B, L, bert_dim]

            # Convert label indices to embeddings (x_0)
            x_0 = self.model.labels_to_embeddings(label_indices)  # [B, L, x_dim]

            # Sample random timesteps
            t = torch.randint(
                0,
                self.model.scheduler.num_steps,
                size=(B,),
                device=self.device,
                dtype=torch.long,
            )  # [B]

            # Sample noise
            noise = torch.randn_like(x_0)  # [B, L, x_dim]

            # Forward diffusion: add noise to x_0
            x_t = self.model.scheduler.q_sample(x_0, t, noise)  # [B, L, x_dim]

            # Predict x_0 from x_t
            x0_pred = self.model.denoiser(
                x_t=x_t,
                t=t,
                token_embeddings=token_embeddings,
                attn_mask=attention_mask.bool(),
            )  # [B, L, x_dim]

            # Compute loss: MSE between predicted and true x_0
            # Only compute loss on non-padding tokens
            mask = attention_mask.unsqueeze(-1).expand_as(x_0)  # [B, L, x_dim]
            loss_per_element = self.criterion(x0_pred, x_0)  # [B, L, x_dim]
            loss = (loss_per_element * mask).sum() / mask.sum()  # Mean over valid tokens

            return {
                "loss": loss.item(),
            }

    def train_epoch(self, dataloader: DataLoader, epoch: int, log_interval: int = 100) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            dataloader: DataLoader for training data
            epoch: Current epoch number
            log_interval: Logging interval
        """
        metrics = super().train_epoch(dataloader, epoch, log_interval)
        metrics.update(self.check_embedding_separation())
        return metrics

    def check_embedding_separation(self) -> Dict[str, float]:
        """
        Check if label embeddings are collapsing.

        Returns:
            Dictionary with embedding separation metrics:
            - min_embedding_similarity: Minimum cosine similarity between any two embeddings
            - mean_embedding_similarity: Mean cosine similarity between all pairs
            - max_embedding_similarity: Maximum cosine similarity (excluding self-similarity)
        """
        with torch.no_grad():
            emb_weights = self.model.label_mapper.embs.weight  # [num_classes, embedding_dim]
            num_classes = emb_weights.shape[0]

            # Normalize embeddings to unit vectors
            normed_emb = torch.nn.functional.normalize(emb_weights, p=2, dim=-1)

            # Compute pairwise cosine similarity matrix
            cosine_sim = torch.matmul(normed_emb, normed_emb.t())  # [num_classes, num_classes]

            # Get off-diagonal similarities (exclude self-similarity)
            mask = ~torch.eye(num_classes, device=cosine_sim.device, dtype=torch.bool)
            off_diagonal_sim = cosine_sim[mask].abs()

            min_sim = off_diagonal_sim.min().item()
            mean_sim = off_diagonal_sim.mean().item()
            max_sim = off_diagonal_sim.max().item()

            return {
                "min_embedding_similarity": min_sim,
                "mean_embedding_similarity": mean_sim,
                "max_embedding_similarity": max_sim,
            }

    def get_checkpoint_state_dict(self) -> Dict[str, torch.Tensor]:
        """Get state dictionaries for checkpointing."""
        return self.model.state_dict(include_encoder=False)

    def load_checkpoint_state_dict(self, checkpoint: Dict[str, torch.Tensor]):
        """Load state dictionaries from checkpoint."""
        self.model.load_state_dict(checkpoint, include_encoder=False)


class DiffusionTrainerConfig(BaseTrainerConfig):
    """
    Configuration model for DiffusionTrainer.
    Inherits from BaseTrainerConfig and adds factory method for DiffusionTrainer.
    """

    def create(self, model: DiffusionSequenceLabeler) -> DiffusionTrainer:
        """
        Factory method to create a DiffusionTrainer instance.

        Args:
            model: Unified diffusion sequence labeler model

        Returns:
            Instance of DiffusionTrainer

        Example:
            config = DiffusionTrainerConfig(learning_rate=2e-4, weight_decay=0.01)
            trainer = config.create_trainer(model=my_model)
        """
        return DiffusionTrainer(
            model=model,
            device=self.device,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            max_grad_norm=self.max_grad_norm,
        )

"""Unified diffusion model for sequence labeling."""

import torch
import torch.nn as nn
from typing import Dict

from diffopenie.diffusion.denoiser import DiffusionSLDenoiser
from diffopenie.diffusion.scheduler import LinearScheduler
from diffopenie.models.label_mapper import LabelMapper
from diffopenie.models.encoder import BERTEncoder


class DiffusionSequenceLabeler(nn.Module):
    """
    Unified diffusion model for sequence labeling.
    
    Encapsulates all components needed for diffusion-based sequence labeling:
    - Encoder: BERT encoder for token embeddings
    - LabelMapper: Maps label indices to embeddings and vice versa
    - Scheduler: Diffusion noise scheduler
    - Denoiser: Denoising model
    
    This class provides a clean interface for both training and inference.
    """
    
    def __init__(
        self,
        denoiser: DiffusionSLDenoiser,
        scheduler: LinearScheduler,
        label_mapper: LabelMapper,
        encoder: BERTEncoder,
    ):
        """
        Args:
            denoiser: Diffusion denoiser model
            scheduler: Diffusion scheduler (handles noise schedule)
            label_mapper: Maps label indices to embeddings
            encoder: BERT encoder for token embeddings
        """
        super().__init__()
        self.denoiser = denoiser
        self.scheduler = scheduler
        self.label_mapper = label_mapper
        self.encoder = encoder
    
    def to(self, device):
        """Move all components to the specified device."""
        super().to(device)
        self.denoiser = self.denoiser.to(device)
        self.scheduler = self.scheduler.to(device)
        self.label_mapper = self.label_mapper.to(device)
        self.encoder = self.encoder.to(device)
        return self
    
    @torch.no_grad()
    def predict(
        self,
        token_ids: torch.LongTensor,  # [B, L]
        attention_mask: torch.Tensor,  # [B, L]
    ) -> torch.LongTensor:
        """
        Perform inference to predict label indices.
        
        Args:
            token_ids: Token IDs [B, L]
            attention_mask: Attention mask [B, L]
        
        Returns:
            Predicted label indices [B, L]
        """
        # Get BERT token embeddings
        token_embeddings = self.encoder(token_ids, attention_mask)  # [B, L, bert_dim]
        
        # Perform step-by-step inference
        noise_shape = (token_ids.shape[0], token_ids.shape[1], self.label_mapper.embedding_dim)
        x0_pred = self.scheduler.inference(
            denoiser=self.denoiser,
            shape=noise_shape,
            condition=token_embeddings,
        )  # [B, L, x_dim]
        
        # Convert predictions back to label indices
        pred_indices = self.label_mapper.reverse(x0_pred)  # [B, L]
        
        return pred_indices
    
    def encode_tokens(
        self,
        token_ids: torch.LongTensor,  # [B, L]
        attention_mask: torch.Tensor,  # [B, L]
    ) -> torch.Tensor:
        """
        Encode tokens using the BERT encoder.
        
        Args:
            token_ids: Token IDs [B, L]
            attention_mask: Attention mask [B, L]
        
        Returns:
            Token embeddings [B, L, bert_dim]
        """
        return self.encoder(token_ids, attention_mask)
    
    def labels_to_embeddings(
        self,
        label_indices: torch.LongTensor,  # [B, L]
    ) -> torch.Tensor:
        """
        Convert label indices to embeddings.
        
        Args:
            label_indices: Label indices [B, L]
        
        Returns:
            Label embeddings [B, L, embedding_dim]
        """
        return self.label_mapper(label_indices)
    
    def embeddings_to_labels(
        self,
        embeddings: torch.Tensor,  # [B, L, embedding_dim]
    ) -> torch.LongTensor:
        """
        Convert embeddings back to label indices.
        
        Args:
            embeddings: Label embeddings [B, L, embedding_dim]
        
        Returns:
            Label indices [B, L]
        """
        return self.label_mapper.reverse(embeddings)
    
    def state_dict(self, include_encoder: bool = False):
        """
        Get state dictionary for checkpointing.
        
        Args:
            include_encoder: Whether to include encoder state (usually frozen)
        
        Returns:
            Dictionary with model state dicts
        """
        state = {
            "denoiser_state_dict": self.denoiser.state_dict(),
            "label_mapper_state_dict": self.label_mapper.state_dict(),
        }
        if include_encoder:
            state["encoder_state_dict"] = self.encoder.state_dict()
        return state
    
    def load_state_dict(self, checkpoint: Dict[str, torch.Tensor], include_encoder: bool = False):
        """
        Load state dictionary from checkpoint.
        
        Args:
            checkpoint: Dictionary with model state dicts
            include_encoder: Whether to load encoder state
        """
        self.denoiser.load_state_dict(checkpoint["denoiser_state_dict"])
        self.label_mapper.load_state_dict(checkpoint["label_mapper_state_dict"])
        if include_encoder and "encoder_state_dict" in checkpoint:
            self.encoder.load_state_dict(checkpoint["encoder_state_dict"])

"""Training module for diffusion-based OpenIE model."""

from .base_trainer import BaseTrainer
from .trainer import DiffusionTrainer

__all__ = ["BaseTrainer", "DiffusionTrainer"]

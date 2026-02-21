"""Training example with configuration and CLI for diffusion-based OpenIE model."""

import argparse
from typing import Annotated, Optional, Union
from torch.utils.data import DataLoader
from pydantic import BaseModel, Field

from diffopenie.utils import load_config
from diffopenie.training.sequence_trainer import DiffusionTrainerConfig
from diffopenie.training.span_trainer_simple import SimpleSpanTrainerConfig
from diffopenie.training.detie_trainer import DetIETrainerConfig
from diffopenie.models.discrete.discrete_trainer import DiscreteTrainerConfig
from diffopenie.models.span import SpanDiffusionModelConfig
from diffopenie.models.sequence import DiffusionSequenceLabelerConfig
from diffopenie.models.detie import DetIEModelConfig
from diffopenie.models.discrete.discrete_model import DiscreteModelConfig
from diffopenie.data.dataset import (
    CachedDatasetConfig,
    SequenceLSOEIDatasetConfig,
    SpanLSOEIDatasetConfig,
)
from diffopenie.data.imojie import SequenceImojieDatasetConfig
from diffopenie.data.collator import SequenceCollator, SpanCollator


DatasetConfigUnion = Annotated[
    SequenceLSOEIDatasetConfig
    | SpanLSOEIDatasetConfig
    | CachedDatasetConfig
    | SequenceImojieDatasetConfig,
    Field(discriminator="type"),
]


def _dataset_type_for_collator(cfg: DatasetConfigUnion) -> str:
    """Return effective dataset type (for cached, use first inner dataset type)."""
    if cfg.type == "cached":
        return cfg.datasets[0].type
    return cfg.type


class DataConfig(BaseModel):
    """Configuration for data loading."""
    train_dataset: DatasetConfigUnion | None = None
    val_dataset: DatasetConfigUnion | None = None
    dataset: DatasetConfigUnion | None = None  # legacy: used for both train and val
    batch_size: int = 32
    num_workers: int = 4
    pad_token_id: int = 0  # not needed for span
    pad_label_idx: int = 0  # not needed for span

    def get_train_config(self) -> DatasetConfigUnion:
        if self.train_dataset is not None:
            return self.train_dataset
        if self.dataset is not None:
            return self.dataset
        raise ValueError("DataConfig must have train_dataset or (legacy) dataset")

    def get_val_config(self) -> DatasetConfigUnion:
        if self.val_dataset is not None:
            return self.val_dataset
        if self.dataset is not None:
            return self.dataset
        raise ValueError("DataConfig must have val_dataset or (legacy) dataset")


class TrainingConfig(BaseModel):
    """
    Complete training configuration.

    Contains:
    - trainer_config: Configuration for the trainer (optimizer, device, etc.)
    - model_config: Configuration for the model (encoder, denoiser, scheduler, etc.)
    - data_config: Configuration for data loading (dataset, dataloader settings)
    """

    trainer: Annotated[
        Union[
            DiffusionTrainerConfig,
            SimpleSpanTrainerConfig,
            DetIETrainerConfig,
            DiscreteTrainerConfig,
        ],
        Field(discriminator="type"),
    ]
    model: (
        DiffusionSequenceLabelerConfig
        | SpanDiffusionModelConfig
        | DetIEModelConfig
        | DiscreteModelConfig
    )
    data: DataConfig

    # Training hyperparameters
    num_epochs: int = 10
    log_interval: int = 100
    save_path: Optional[str] = None
    save_interval: int = 1
    val_full_interval: int = 5
    num_classes: int = 4  # O, Subject, Object, Predicate


def create_training_components(config: TrainingConfig):
    """
    Create all components needed for training from configuration.

    Args:
        config: TrainingConfig instance

    Returns:
        Dictionary containing:
            - model: DiffusionSequenceLabeler instance
            - trainer: DiffusionTrainer instance
            - train_dataloader: DataLoader for training
            - val_dataloader: DataLoader for validation (or None)
    """
    # Create model
    model = config.model.create()

    # Create trainer
    trainer = config.trainer.create(model=model)

    # Create datasets (split is set in each dataset config, not passed to create())
    train_cfg = config.data.get_train_config()
    val_cfg = config.data.get_val_config()
    train_dataset = train_cfg.create()
    val_dataset = val_cfg.create()

    # Collator from train dataset type (cached -> first inner type)
    dataset_type = _dataset_type_for_collator(train_cfg)
    if dataset_type in ("sequence", "imojie"):
        collator = SequenceCollator(
            pad_token_id=config.data.pad_token_id,
            pad_label_idx=config.data.pad_label_idx,
        )
    elif dataset_type == "span":
        collator = SpanCollator(
            pad_token_id=config.data.pad_token_id,
        )
    else:
        raise ValueError(f"Invalid dataset type: {dataset_type}")

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        collate_fn=collator,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        collate_fn=collator,
    )

    return {
        "model": model,
        "trainer": trainer,
        "train_dataloader": train_dataloader,
        "val_dataloader": val_dataloader,
    }


def main():
    """Main entry point for training script."""
    parser = argparse.ArgumentParser(
        description="Train diffusion-based OpenIE model using YAML configuration"
    )
    parser.add_argument(
        "config_path",
        type=str,
        help="Path to YAML configuration file containing TrainingConfig",
    )

    args = parser.parse_args()

    # Load configuration from YAML file
    config = load_config(TrainingConfig, args.config_path)

    print(f"Loaded configuration from {args.config_path}")
    print(f"Training for {config.num_epochs} epochs")
    print(f"Batch size: {config.data.batch_size}")
    print(f"Device: {config.trainer.device}")

    # Create training components
    components = create_training_components(config)

    print("Training components created successfully!")
    print(f"Model: {components['model']}")
    print(f"Trainer: {components['trainer']}")
    print(f"Train batches: {len(components['train_dataloader'])}")
    print(f"Val batches: {len(components['val_dataloader'])}")
    print("-" * 80)

    # Start training
    components["trainer"].train(
        train_dataloader=components["train_dataloader"],
        num_epochs=config.num_epochs,
        log_interval=config.log_interval,
        save_path=config.save_path,
        save_interval=config.save_interval,
        val_dataloader=components["val_dataloader"],
        num_classes=config.num_classes,
        val_full_interval=config.val_full_interval,
    )

    print("-" * 80)
    print("Training completed!")


if __name__ == "__main__":
    main()

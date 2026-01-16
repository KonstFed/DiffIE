"""Training example with configuration and CLI for diffusion-based OpenIE model."""

import argparse
from typing import Optional
from torch.utils.data import DataLoader
from pydantic import BaseModel

from diffopenie.utils import load_config
from diffopenie.training.trainer import DiffusionTrainerConfig
from diffopenie.models.diffusion_model import DiffusionSequenceLabelerConfig
from diffopenie.data.dataset import SequenceLSOEIDataset
from diffopenie.data.collator import DiffusionCollator


class DataConfig(BaseModel):
    """Configuration for data loading."""

    dataset_name: str = "SequenceLSOEIDataset"  # Dataset class name
    tokenizer_name: str = "bert-base-uncased"
    batch_size: int = 32
    num_workers: int = 4
    pad_token_id: int = 0
    pad_label_idx: int = 0


class TrainingConfig(BaseModel):
    """
    Complete training configuration.

    Contains:
    - trainer_config: Configuration for the trainer (optimizer, device, etc.)
    - model_config: Configuration for the diffusion model (encoder, denoiser, scheduler, etc.)
    - data_config: Configuration for data loading (dataset, dataloader settings)
    """

    trainer: DiffusionTrainerConfig
    model: DiffusionSequenceLabelerConfig
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

    # Create datasets
    train_dataset = SequenceLSOEIDataset(
        split="train",
        tokenizer_name=config.data.tokenizer_name,
    )

    val_dataset = SequenceLSOEIDataset(
        split="validation",
        tokenizer_name=config.data.tokenizer_name,
    )

    # Create collator
    collator = DiffusionCollator(
        pad_token_id=config.data.pad_token_id,
        pad_label_idx=config.data.pad_label_idx,
    )

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

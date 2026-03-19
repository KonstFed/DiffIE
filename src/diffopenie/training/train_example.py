"""CLI entry point for training discrete diffusion OpenIE model."""

import argparse
from pathlib import Path
from typing import Annotated, Optional

import torch
from pydantic import BaseModel, ConfigDict, Field
from torch.utils.data import DataLoader

from diffopenie.data import SEQ_STR2INT
from diffopenie.data.collator import SequenceCollator, SpanCollator
from diffopenie.data.dataset import (
    CachedDatasetConfig,
    SequenceLSOEIDatasetConfig,
    SpanLSOEIDatasetConfig,
)
from diffopenie.data.imojie import SequenceImojieDatasetConfig
from diffopenie.models.discrete.discrete_model import DiscreteModelConfig
from diffopenie.training.trainer import Trainer, TrainerConfig
from diffopenie.utils import load_config

DatasetConfigUnion = Annotated[
    SequenceLSOEIDatasetConfig
    | SpanLSOEIDatasetConfig
    | CachedDatasetConfig
    | SequenceImojieDatasetConfig,
    Field(discriminator="type"),
]


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    train_dataset: DatasetConfigUnion | None = None
    val_dataset: DatasetConfigUnion | None = None
    dataset: DatasetConfigUnion | None = None  # legacy fallback
    batch_size: int = 32
    num_workers: int = 4
    pad_token_id: int = 0
    pad_label_idx: int = SEQ_STR2INT["PAD"]

    def get_train_config(self) -> DatasetConfigUnion:
        if self.train_dataset is not None:
            return self.train_dataset
        if self.dataset is not None:
            return self.dataset
        raise ValueError("No train dataset configured")

    def get_val_config(self) -> DatasetConfigUnion:
        if self.val_dataset is not None:
            return self.val_dataset
        if self.dataset is not None:
            return self.dataset
        raise ValueError("No val dataset configured")


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trainer: TrainerConfig
    model: DiscreteModelConfig
    data: DataConfig

    num_epochs: int = 10
    log_interval: int = 100  # kept for YAML backward compat
    save_path: Optional[str] = None
    save_interval: int = 1
    val_full_interval: int = 5
    val_metrics_on_train: bool = False
    train_val_batches: Optional[int] = None
    model_weights: Optional[str] = None  # path to checkpoint for weight init (pretrain→finetune)


def _collator_for(cfg: DatasetConfigUnion, data: DataConfig):
    dtype = cfg.datasets[0].type if cfg.type == "cached" else cfg.type
    if dtype in ("sequence", "imojie"):
        return SequenceCollator(
            pad_token_id=data.pad_token_id,
            pad_label_idx=data.pad_label_idx,
        )
    if dtype == "span":
        return SpanCollator(pad_token_id=data.pad_token_id)
    raise ValueError(f"Unknown dataset type: {dtype}")


def create_training_components(
    config: TrainingConfig,
) -> tuple[DiscreteModelConfig, Trainer, DataLoader, DataLoader]:
    model = config.model.create()

    train_cfg = config.data.get_train_config()
    val_cfg = config.data.get_val_config()
    collator = _collator_for(train_cfg, config.data)

    train_ds = train_cfg.create()
    train_dl = DataLoader(
        train_ds,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        collate_fn=collator,
    )
    val_dl = DataLoader(
        val_cfg.create(),
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        collate_fn=collator,
    )

    # Compute total training steps for LR scheduler
    steps_per_epoch = len(train_dl)
    total_steps = steps_per_epoch * config.num_epochs
    trainer = config.trainer.create(model=model, total_steps=total_steps)

    if config.model_weights is not None:
        ckpt = torch.load(config.model_weights, map_location=trainer.device)
        state = ckpt.get("model_state_dict", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"[model_weights] missing keys: {missing}")
        if unexpected:
            print(f"[model_weights] unexpected keys: {unexpected}")
        print(f"[model_weights] loaded from {config.model_weights}")

    return model, trainer, train_dl, val_dl


def main():
    parser = argparse.ArgumentParser(
        description="Train discrete diffusion OpenIE model",
    )
    parser.add_argument("config_path", type=str)
    parser.add_argument("--log-path", type=str, default=None)
    args = parser.parse_args()

    config = load_config(TrainingConfig, args.config_path)
    log_path = args.log_path or f"{Path(args.config_path).stem}.csv"

    _model, trainer, train_dl, val_dl = create_training_components(config)

    print(f"Config: {args.config_path}")
    print(f"Train: {len(train_dl)} batches, Val: {len(val_dl)} batches")

    trainer.train(
        train_dataloader=train_dl,
        num_epochs=config.num_epochs,
        save_path=config.save_path,
        save_interval=config.save_interval,
        val_dataloader=val_dl,
        val_full_interval=config.val_full_interval,
        val_metrics_on_train=config.val_metrics_on_train,
        log_path=log_path,
        train_val_batches=config.train_val_batches,
        carb_gold_path=config.trainer.carb_gold_path,
        carb_sentences_path=config.trainer.carb_sentences_path,
    )
    print("Training completed!")


if __name__ == "__main__":
    main()

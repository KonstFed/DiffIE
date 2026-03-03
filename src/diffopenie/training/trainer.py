"""Discrete diffusion trainer (merged from BaseTrainer + DiscreteTrainer)."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
from pydantic import BaseModel, ConfigDict, model_validator
from torch.utils.data import DataLoader
from tqdm import tqdm

from diffopenie.data import SEQ_STR2INT
from diffopenie.models.discrete.discrete_model import DiscreteModel
from diffopenie.utils import hprint
from diffopenie.training.logger import TrainingLogger
from diffopenie.training.metrics import (
    EpochResult,
    MetricsResult,
    PerTimestepLoss,
    PerTimestepMetricsResult,
    PerTimestepTripletMetrics,
    TripletMetrics,
    ValidationResult,
)

DEFAULT_CLASS_WEIGHTS = [0.25, 0.25, 0.25, 0.25]
IGNORE_INDEX = -100


@dataclass
class ForwardResult:
    """Output of a single forward pass (noising + denoising)."""

    loss: torch.Tensor
    per_sample_loss: torch.Tensor  # [B]
    predictions: torch.Tensor  # [B, L]
    labels: torch.Tensor  # [B, L]
    mask: torch.Tensor  # [B, L]
    timesteps: torch.Tensor  # [B]


class Trainer:
    def __init__(
        self,
        model: DiscreteModel,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        class_weights: list[float] | None = None,
        background_drop_prob: float = 0.8,
    ):
        self.model = model.to(device)
        self.model.scheduler.to(device)
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.background_drop_prob = background_drop_prob
        self.global_step = 0

        weights = list(class_weights or DEFAULT_CLASS_WEIGHTS)
        if self.model.scheduler.kernel == "mask_absorbing":
            weights.append(1e10)

        self._loss_fn = nn.CrossEntropyLoss(
            reduction="none",
            ignore_index=IGNORE_INDEX,
            weight=torch.tensor(weights, dtype=torch.float32, device=device),
        )
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay,
        )

    def _to_device(self, batch: dict[str, torch.Tensor]):
        return (
            batch["token_ids"].to(self.device),
            batch["attention_mask"].to(self.device),
            batch["label_indices"].to(self.device),
        )

    # -- Loss computation (trainer-specific: masking, bg drop, per-t loss) -

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> ForwardResult:
        """Noise labels, denoise, compute masked CE loss."""
        token_ids, attention_mask, labels = self._to_device(batch)
        B, L = token_ids.shape

        # # DEBUG: convert token_ids to tokens (first sample only) and print S/O/R with colour
        # tokenizer = self.model.encoder.tokenizer
        # first_ids = token_ids[0].cpu().tolist()
        # tokens = tokenizer.convert_ids_to_tokens(first_ids)
        # print("[DEBUG] token_ids[0] -> tokens:", tokens)
        # first_labels = labels[0].cpu().tolist()
        # subject_ind = [i for i, lb in enumerate(first_labels) if lb == SEQ_STR2INT["S"]]
        # object_ind = [i for i, lb in enumerate(first_labels) if lb == SEQ_STR2INT["O"]]
        # relation_ind = [i for i, lb in enumerate(first_labels) if lb == SEQ_STR2INT["R"]]
        # print("[DEBUG] subject/object/relation tokens:")
        # hprint(tokens, subject_ind, object_ind, relation_ind, legend=True)

        token_emb = self.model.encode_tokens(token_ids, attention_mask)
        t = self.model.scheduler.sample_t(B)
        x_t = self.model.noise(labels, t)
        logits = self.model.denoiser(x_t, t, token_emb, attention_mask)
        predictions = logits.argmax(dim=-1)

        target = labels.clone()
        target[attention_mask == 0] = IGNORE_INDEX
        if self.model.scheduler.kernel == "mask_absorbing":
            target[x_t != self.model.scheduler.mask_state_id] = IGNORE_INDEX

        if self.background_drop_prob > 0 and self.model.training:
            bg_and_active = (labels == SEQ_STR2INT["B"]) & (target != IGNORE_INDEX)
            drop = bg_and_active & (
                torch.rand_like(target, dtype=torch.float32)
                < self.background_drop_prob
            )
            target[drop] = IGNORE_INDEX

        per_token = self._loss_fn(
            logits.reshape(-1, logits.size(-1)), target.reshape(-1),
        ).reshape(B, L)
        valid = (target != IGNORE_INDEX).float()
        per_sample_loss = (per_token * valid).sum(1) / valid.sum(1).clamp(min=1)
        loss = per_sample_loss.mean()

        metric_mask = attention_mask.clone()
        metric_mask[target == IGNORE_INDEX] = 0

        return ForwardResult(
            loss=loss,
            per_sample_loss=per_sample_loss,
            predictions=predictions,
            labels=labels,
            mask=metric_mask,
            timesteps=t,
        )

    # -- Epoch-level methods ----------------------------------------------

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> EpochResult:
        self.model.train()
        metrics = TripletMetrics().to(self.device)
        per_t = PerTimestepLoss(self.model.scheduler.num_steps).to(self.device)
        total_loss, n = 0.0, 0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch}", leave=False):
            result = self.compute_loss(batch)
            result.loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.max_grad_norm,
            )
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.global_step += 1

            total_loss += result.loss.item()
            n += 1
            metrics.update(result.predictions, result.labels, result.mask)
            per_t.update(result.per_sample_loss, result.timesteps)

        return EpochResult(
            loss=total_loss / max(n, 1),
            direct_metrics=metrics.compute(),
            per_timestep_loss=per_t.compute(),
            t_sampled_counts=per_t.get_counts(),
        )

    @torch.no_grad()
    def validate_loss(self, dataloader: DataLoader) -> float:
        self.model.eval()
        total, n = 0.0, 0
        for batch in tqdm(dataloader, desc="Val loss", leave=False):
            total += self.compute_loss(batch).loss.item()
            n += 1
        return total / max(n, 1)

    @torch.no_grad()
    def validate_per_t_loss(
        self, dataloader: DataLoader
    ) -> torch.Tensor | None:
        """Average loss per timestep on the validation set. Returns [T] or None if empty."""
        self.model.eval()
        per_t = PerTimestepLoss(self.model.scheduler.num_steps).to(self.device)
        n = 0
        for batch in tqdm(dataloader, desc="Val per-t loss", leave=False):
            result = self.compute_loss(batch)
            per_t.update(result.per_sample_loss, result.timesteps)
            n += 1
        if n == 0:
            return None
        return per_t.compute()

    @torch.no_grad()
    def validate(
        self,
        dataloader: DataLoader,
        max_batches: int | None = None,
        compute_per_timestep_metrics: bool = False,
    ) -> ValidationResult:
        self.model.eval()
        metrics = TripletMetrics().to(self.device)
        num_steps = self.model.scheduler.num_steps
        mask_state_id = (
            self.model.scheduler.mask_state_id
            if getattr(self.model.scheduler, "kernel", None) == "mask_absorbing"
            else None
        )
        per_t_metrics = (
            PerTimestepTripletMetrics(
                num_steps, mask_state_id=mask_state_id
            ).to(self.device)
            if compute_per_timestep_metrics
            else None
        )
        for i, batch in enumerate(
            tqdm(dataloader, desc="Validating", leave=False),
        ):
            if max_batches is not None and i >= max_batches:
                break
            token_ids, attention_mask, labels = self._to_device(batch)
            token_emb = self.model.encode_tokens(token_ids, attention_mask)
            out = self.model.generate(
                batch_size=token_ids.shape[0],
                token_embeddings=token_emb,
                attention_mask=attention_mask,
                return_intermediate=compute_per_timestep_metrics,
            )
            if compute_per_timestep_metrics:
                preds, intermediates = out
                metrics.update(preds, labels, attention_mask)
                per_t_metrics.update(intermediates, labels, attention_mask)
            else:
                metrics.update(out, labels, attention_mask)
        carb = metrics.compute()
        per_t_carb = per_t_metrics.compute() if per_t_metrics is not None else None
        return ValidationResult(carb=carb, per_t_carb=per_t_carb)

    # -- Main training loop -----------------------------------------------

    def train(
        self,
        train_dataloader: DataLoader,
        num_epochs: int = 11,
        save_path: str | None = None,
        save_interval: int = 1,
        val_dataloader: DataLoader | None = None,
        val_full_interval: int = 5,
        val_metrics_on_train: bool = False,
        log_path: str | None = None,
        train_val_batches: int | None = None,
    ):
        log_resolved = (
            Path(log_path) if log_path
            else Path(save_path) / "train_log.csv" if save_path
            else None
        )
        logger = TrainingLogger(log_resolved)
        best_f1 = -1.0

        params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print(
            f"Training {num_epochs} epochs on {self.device} "
            f"| {params:,} trainable params"
        )

        for epoch in range(1, num_epochs + 1):
            epoch_result = self.train_epoch(train_dataloader, epoch)

            val_loss = (
                self.validate_loss(val_dataloader)
                if val_dataloader else None
            )
            per_t_val_loss = (
                self.validate_per_t_loss(val_dataloader)
                if val_dataloader else None
            )

            do_full = epoch % val_full_interval == 0
            val_result = (
                self.validate(
                    val_dataloader,
                    compute_per_timestep_metrics=do_full,
                )
                if val_dataloader and do_full
                else None
            )
            carb = val_result.carb if val_result else None
            per_t_carb = val_result.per_t_carb if val_result else None
            train_val_result = (
                self.validate(
                    train_dataloader,
                    max_batches=train_val_batches,
                    compute_per_timestep_metrics=do_full,
                )
                if val_metrics_on_train and do_full
                else None
            )
            train_carb = train_val_result.carb if train_val_result else None
            train_per_t_carb = (
                train_val_result.per_t_carb if train_val_result else None
            )

            new_best = None
            if carb and save_path and carb.f1 > best_f1:
                best_f1 = carb.f1
                self.save_checkpoint(save_path, epoch, suffix="best")
                new_best = best_f1

            logger.log_epoch(
                epoch, epoch_result.loss, val_loss,
                epoch_result.direct_metrics, carb, train_carb,
                epoch_result.per_timestep_loss,
                per_t_val_loss=per_t_val_loss,
                t_sampled_counts=epoch_result.t_sampled_counts,
                per_t_carb_metrics=per_t_carb,
                train_per_t_carb_metrics=train_per_t_carb,
            )
            logger.print_epoch(
                epoch, num_epochs, epoch_result.loss, val_loss,
                epoch_result.direct_metrics, carb, train_carb, new_best,
            )

            if save_path and epoch % save_interval == 0:
                self.save_checkpoint(save_path, epoch)

    # -- Checkpointing ----------------------------------------------------

    def save_checkpoint(
        self, path: str, epoch: int, suffix: str | None = None,
    ):
        os.makedirs(path, exist_ok=True)
        fname = (
            f"{path}/checkpoint_{suffix}.pt" if suffix
            else f"{path}/checkpoint_epoch_{epoch}.pt"
        )
        torch.save(
            {
                "epoch": epoch,
                "global_step": self.global_step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            fname,
        )
        print(f"Saved {fname}")

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        state = ckpt.get("model_state_dict", ckpt)
        self.model.load_state_dict(state, strict=False)
        self.global_step = ckpt.get("global_step", 0)
        if "optimizer_state_dict" in ckpt and ckpt["optimizer_state_dict"]:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        print(f"Loaded {path}")


# -- Config ---------------------------------------------------------------


class TrainerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["discrete_trainer"] = "discrete_trainer"
    device: str | None = None
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    background_drop_prob: float = 0.8

    @model_validator(mode="after")
    def _auto_device(self):
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        return self

    def create(self, model: DiscreteModel) -> Trainer:
        return Trainer(
            model=model,
            device=self.device,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            max_grad_norm=self.max_grad_norm,
            background_drop_prob=self.background_drop_prob,
        )

"""Discrete diffusion trainer (merged from BaseTrainer + DiscreteTrainer)."""

import copy
import math
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
from pydantic import BaseModel, ConfigDict, model_validator
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from diffopenie.data import SEQ_STR2INT
from diffopenie.models.discrete.discrete_model import DiscreteModel
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
from diffopenie.utils import hprint

# DEFAULT_CLASS_WEIGHTS = [0.25, 0.25, 0.25, 0.25]
DEFAULT_CLASS_WEIGHTS = [0.1, 0.3, 0.3, 0.3]

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
        encoder_lr: float | None = None,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        class_weights: list[float] | None = None,
        background_drop_prob: float = 0.8,
        label_smoothing: float = 0.0,
        warmup_steps: int = 0,
        total_steps: int | None = None,
        ema_decay: float = 0.0,
    ):
        self.model = model.to(device)
        self.model.scheduler.to(device)
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.background_drop_prob = background_drop_prob
        self.global_step = 0

        # EMA shadow weights
        self.ema_decay = ema_decay
        if ema_decay > 0:
            self._ema_state = copy.deepcopy(model.state_dict())
        else:
            self._ema_state = None

        weights = list(class_weights or DEFAULT_CLASS_WEIGHTS)
        if self.model.scheduler.kernel == "mask_absorbing":
            weights.append(float("inf"))

        self._loss_fn = nn.CrossEntropyLoss(
            reduction="none",
            ignore_index=IGNORE_INDEX,
            weight=torch.tensor(weights, dtype=torch.float32, device=device),
            label_smoothing=label_smoothing,
        )

        # Separate parameter groups for encoder vs denoiser
        encoder_params = list(model.encoder.parameters())
        encoder_ids = {id(p) for p in encoder_params}
        denoiser_params = [p for p in model.parameters() if id(p) not in encoder_ids]
        enc_lr = encoder_lr if encoder_lr is not None else learning_rate

        self.optimizer = torch.optim.AdamW(
            [
                {"params": encoder_params, "lr": enc_lr},
                {"params": denoiser_params, "lr": learning_rate},
            ],
            weight_decay=weight_decay,
        )

        # LR scheduler: linear warmup then cosine decay
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        if warmup_steps > 0 and total_steps is not None and total_steps > 0:
            self.lr_scheduler = LambdaLR(
                self.optimizer,
                lr_lambda=self._warmup_cosine_lambda,
            )
        else:
            self.lr_scheduler = None

    # -- EMA helpers --------------------------------------------------------

    def _update_ema(self):
        if self._ema_state is None:
            return
        d = self.ema_decay
        for k, v in self.model.state_dict().items():
            self._ema_state[k].lerp_(v, 1.0 - d)

    @contextmanager
    def ema_scope(self):
        """Temporarily swap EMA weights into the model for eval."""
        if self._ema_state is None:
            yield
            return
        live = copy.deepcopy(self.model.state_dict())
        self.model.load_state_dict(self._ema_state, strict=False)
        try:
            yield
        finally:
            self.model.load_state_dict(live, strict=False)

    def _warmup_cosine_lambda(self, step: int) -> float:
        if step < self.warmup_steps:
            return step / max(self.warmup_steps, 1)
        progress = (step - self.warmup_steps) / max(
            self.total_steps - self.warmup_steps, 1
        )
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

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
                torch.rand_like(target, dtype=torch.float32) < self.background_drop_prob
            )
            target[drop] = IGNORE_INDEX

        per_token = self._loss_fn(
            logits.reshape(-1, logits.size(-1)),
            target.reshape(-1),
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
                self.model.parameters(),
                self.max_grad_norm,
            )
            self.optimizer.step()
            self._update_ema()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
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
    def validate_per_t_loss(self, dataloader: DataLoader) -> torch.Tensor | None:
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
            PerTimestepTripletMetrics(num_steps, mask_state_id=mask_state_id).to(
                self.device
            )
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

    @torch.no_grad()
    def validate_carb(
        self,
        sentences: list[str],
        gold: dict[str, list],
    ):
        """Run CaRB evaluation using model.get_carb_prediction.

        Uses model.carb_k and model.carb_topk for sampling parameters.

        Args:
            sentences: raw sentences (one per line from CaRB dev/test.txt)
            gold: gold extractions dict (from load_gold_file on CaRB gold/*.tsv)
        """
        from diffopenie.evaluation.carb_metrics import (
            Extraction,
        )
        from diffopenie.evaluation.carb_metrics import (
            evaluate as carb_evaluate,
        )

        self.model.eval()
        predicted: dict[str, list] = {}

        for sent in tqdm(sentences, desc="CaRB val", leave=False):
            words = sent.split()
            triplets, probs = self.model.get_carb_prediction(
                words, k=self.model.carb_k, topk=self.model.carb_topk
            )
            exs = []
            for (sub_span, obj_span, pred_span), prob in zip(triplets, probs):
                subj = " ".join(words[sub_span[0] : sub_span[1] + 1])
                obj_ = " ".join(words[obj_span[0] : obj_span[1] + 1])
                pred = " ".join(words[pred_span[0] : pred_span[1] + 1])
                if subj and obj_ and pred:
                    exs.append(
                        Extraction(pred=pred, args=[subj, obj_], confidence=prob)
                    )
            if exs:
                predicted[sent] = exs

        return carb_evaluate(gold, predicted)

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
        carb_gold_path: str | None = None,
        carb_sentences_path: str | None = None,
        val_lsoie: bool = True,
        val_carb: bool = True,
    ):
        from diffopenie.evaluation.carb_metrics import (
            CarbResult,
            load_gold_file,
        )

        log_resolved = (
            Path(log_path)
            if log_path
            else Path(save_path) / "train_log.csv"
            if save_path
            else None
        )
        logger = TrainingLogger(log_resolved)
        best_f1 = -1.0

        # Load CaRB gold data once if enabled and paths provided
        carb_gold: dict | None = None
        carb_sentences: list[str] | None = None
        if val_carb and carb_gold_path and carb_sentences_path:
            carb_gold = load_gold_file(carb_gold_path)
            with open(carb_sentences_path) as f:
                carb_sentences = [line.strip() for line in f if line.strip()]
            print(
                f"CaRB val: {len(carb_sentences)} sentences, "
                f"{sum(len(v) for v in carb_gold.values())} "
                f"gold extractions"
            )

        params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(
            f"Training {num_epochs} epochs on {self.device} "
            f"| {params:,} trainable params"
        )

        for epoch in range(1, num_epochs + 1):
            epoch_result = self.train_epoch(train_dataloader, epoch)

            with self.ema_scope():
                val_loss = (
                    self.validate_loss(val_dataloader) if val_dataloader else None
                )
                per_t_val_loss = (
                    self.validate_per_t_loss(val_dataloader) if val_dataloader else None
                )

                do_full = epoch % val_full_interval == 0

                # -- LSOIE token-overlap validation --
                carb = None
                per_t_carb = None
                train_carb = None
                train_per_t_carb = None
                if val_lsoie and val_dataloader and do_full:
                    val_result = self.validate(
                        val_dataloader,
                        compute_per_timestep_metrics=do_full,
                    )
                    carb = val_result.carb
                    per_t_carb = val_result.per_t_carb
                    if val_metrics_on_train:
                        train_val_result = self.validate(
                            train_dataloader,
                            max_batches=train_val_batches,
                            compute_per_timestep_metrics=do_full,
                        )
                        train_carb = train_val_result.carb
                        train_per_t_carb = train_val_result.per_t_carb

                # -- CaRB benchmark validation --
                carb_result: CarbResult | None = None
                if carb_gold and carb_sentences and do_full:
                    carb_result = self.validate_carb(
                        carb_sentences,
                        carb_gold,
                    )

            # Best model: prefer CaRB F1, fall back to LSOIE F1
            new_best = None
            current_f1 = None
            if carb_result:
                current_f1 = carb_result.f1
            elif carb:
                current_f1 = carb.f1
            if current_f1 is not None and save_path and current_f1 > best_f1:
                best_f1 = current_f1
                self.save_checkpoint(save_path, epoch, suffix="best")
                new_best = best_f1

            logger.log_epoch(
                epoch,
                epoch_result.loss,
                val_loss,
                epoch_result.direct_metrics,
                carb_metrics=carb,
                train_carb_metrics=train_carb,
                per_t_loss=epoch_result.per_timestep_loss,
                per_t_val_loss=per_t_val_loss,
                t_sampled_counts=epoch_result.t_sampled_counts,
                per_t_carb_metrics=per_t_carb,
                train_per_t_carb_metrics=train_per_t_carb,
                carb_result=carb_result,
            )
            logger.print_epoch(
                epoch,
                num_epochs,
                epoch_result.loss,
                val_loss,
                epoch_result.direct_metrics,
                carb_metrics=carb,
                train_carb_metrics=train_carb,
                new_best=new_best,
                carb_result=carb_result,
            )

            if save_path and epoch % save_interval == 0:
                self.save_checkpoint(save_path, epoch)

    # -- Checkpointing ----------------------------------------------------

    def save_checkpoint(
        self,
        path: str,
        epoch: int,
        suffix: str | None = None,
    ):
        os.makedirs(path, exist_ok=True)
        fname = (
            f"{path}/checkpoint_{suffix}.pt"
            if suffix
            else f"{path}/checkpoint_epoch_{epoch}.pt"
        )
        # For "best" checkpoint, save EMA weights as model_state_dict so
        # load_checkpoint (used in eval) gets the smoothed weights directly.
        if suffix == "best" and self._ema_state is not None:
            model_state = self._ema_state
        else:
            model_state = self.model.state_dict()

        torch.save(
            {
                "epoch": epoch,
                "global_step": self.global_step,
                "model_state_dict": model_state,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "ema_state_dict": self._ema_state,
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
        if "ema_state_dict" in ckpt and ckpt["ema_state_dict"] is not None:
            self._ema_state = ckpt["ema_state_dict"]
        print(f"Loaded {path}")


# -- Config ---------------------------------------------------------------


class TrainerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["discrete_trainer"] = "discrete_trainer"
    device: str | None = None
    learning_rate: float = 1e-4
    encoder_lr: float | None = None
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    background_drop_prob: float = 0.8
    label_smoothing: float = 0.0
    warmup_steps: int = 0
    ema_decay: float = 0.0  # 0 = disabled; typical values: 0.999, 0.9999

    # CaRB benchmark validation
    carb_gold_path: str | None = None  # path to CaRB gold/*.tsv
    carb_sentences_path: str | None = None  # path to CaRB dev.txt/test.txt

    @model_validator(mode="after")
    def _auto_device(self):
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        return self

    def create(self, model: DiscreteModel, total_steps: int | None = None) -> Trainer:
        return Trainer(
            model=model,
            device=self.device,
            learning_rate=self.learning_rate,
            encoder_lr=self.encoder_lr,
            weight_decay=self.weight_decay,
            max_grad_norm=self.max_grad_norm,
            background_drop_prob=self.background_drop_prob,
            label_smoothing=self.label_smoothing,
            warmup_steps=self.warmup_steps,
            total_steps=total_steps,
            ema_decay=self.ema_decay,
        )

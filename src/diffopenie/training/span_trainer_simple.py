"""Trainer for span diffusion OpenIE model."""
from typing import Dict, List, Literal

import torch
import torch.nn as nn

from diffopenie.models.span import SpanDiffusionModel
from diffopenie.models.span.span_model import spans_to_token_labels
from diffopenie.training.base_trainer import BaseTrainer, BaseTrainerConfig


class SimpleSpanTrainer(BaseTrainer):
    """Simple Trainer for SpanDiffusionModel; same metrics/logs as base/sequence (CaRB).

    Assumes that label mapper outputs single constant length vector for each item.
    Denoiser model outputs vector of same size. Pure DDPM style.
    """

    def __init__(
        self,
        model: SpanDiffusionModel,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
    ):
        self.model = model.to(device)
        super().__init__(
            device=device,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
        )
        self.criterion = nn.MSELoss(reduction="mean")

    def get_trainable_models(self) -> List[nn.Module]:
        return [self.model.denoiser]

    def get_eval_models(self) -> List[nn.Module]:
        return [self.model.scheduler, self.model.denoiser]


    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        token_ids = batch["token_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        label_spans = batch["label_spans"].to(self.device)
        seq_len = batch["seq_len"].to(self.device)
        B, L = token_ids.shape

        token_embeddings = batch["token_embeddings"].to(self.device)
        x_0 = self.model.label_mapper.forward(label_spans, seq_len)
        t = torch.randint(
            0,
            self.model.scheduler.num_steps,
            size=(B,),
            device=self.device,
            dtype=torch.long,
        )
        x_t = self.model.noise(x_0, t)

        x0_pred = self.model.denoise(
            x_t=x_t,
            t=t,
            token_embeddings=token_embeddings,
            attention_mask=attention_mask.bool(),
        )

        # predict x0 can switch to predict noise?
        loss = self.criterion(x0_pred, x_0)

        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        self.global_step += 1
        return {"loss": loss.item()}

    def validate_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            attention_mask = batch["attention_mask"].to(self.device)
            label_spans = batch["label_spans"].to(self.device)
            B, L = attention_mask.shape
            token_embeddings = batch["token_embeddings"].to(self.device)
            pred_spans = self.model.predict_from_embeddings(
                token_embeddings, attention_mask
            )
            pred_labels = spans_to_token_labels(pred_spans, L)
            gold_labels = spans_to_token_labels(label_spans, L)
            return {
                "predictions": pred_labels,
                "labels": gold_labels,
                "attention_mask": attention_mask,
            }

    def validate_loss_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        with torch.no_grad():
            token_ids = batch["token_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            label_spans = batch["label_spans"].to(self.device)
            seq_len = batch["seq_len"].to(self.device)
            token_embeddings = batch["token_embeddings"].to(self.device)
            B, L = token_ids.shape
            x_0 = self.model.label_mapper.forward(label_spans, seq_len)
            t = torch.randint(
                0,
                self.model.scheduler.num_steps,
                size=(B,),
                device=self.device,
                dtype=torch.long,
            )
            x_t = self.model.noise(x_0, t)
            x0_pred = self.model.denoise(
                x_t=x_t,
                t=t,
                token_embeddings=token_embeddings,
                attention_mask=attention_mask.bool(),
            )
            loss = self.criterion(x0_pred, x_0)
            return {"loss": loss.item()}

    def get_checkpoint_state_dict(self) -> Dict[str, torch.Tensor]:
        sd = self.model.state_dict()
        return {k: v for k, v in sd.items() if not k.startswith("encoder.")}

    def load_checkpoint_state_dict(self, checkpoint: Dict[str, torch.Tensor]):
        to_load = {k: v for k, v in checkpoint.items() if not k.startswith("encoder.")}
        self.model.load_state_dict(to_load, strict=False)


class SimpleSpanTrainerConfig(BaseTrainerConfig):
    type: Literal["span_trainer"] = "span_trainer"

    def create(self, model: SpanDiffusionModel) -> SimpleSpanTrainer:
        return SimpleSpanTrainer(
            model=model,
            device=self.device,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            max_grad_norm=self.max_grad_norm,
        )

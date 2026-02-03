"""Trainer for span diffusion OpenIE model."""
from typing import Dict, List, Literal

import torch
import torch.nn as nn

from diffopenie.models.span import SpanDiffusionModel
from diffopenie.models.span.span_model import spans_to_token_labels
from diffopenie.training.base_trainer import BaseTrainer, BaseTrainerConfig


class SpanDiffusionTrainer(BaseTrainer):
    """Trainer for SpanDiffusionModel; same metrics/logs as base/sequence (CaRB)."""

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
        self.criterion = nn.MSELoss(reduction="none")

    def get_trainable_models(self) -> List[nn.Module]:
        return [self.model.denoiser]

    def get_eval_models(self) -> List[nn.Module]:
        return [self.model.scheduler, self.model.encoder]

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        token_ids = batch["token_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        label_spans = batch["label_spans"].to(self.device)
        B, L = token_ids.shape

        token_embeddings = self.model.encode_tokens(token_ids, attention_mask)
        x_0 = self.model.label_mapper.forward(label_spans, L).reshape(B, 6, L)
        t = torch.randint(
            0,
            self.model.scheduler.num_steps,
            size=(B,),
            device=self.device,
            dtype=torch.long,
        )
        x_t = self.model.noise(x_0, t)

        x0_pred = self.model.denoiser(
            x_t=x_t,
            t=t,
            condition=(token_embeddings, attention_mask.bool()),
        )
        mask = attention_mask.unsqueeze(1).expand_as(x_0)
        loss = (self.criterion(x0_pred, x_0) * mask).sum() / mask.sum().clamp(min=1)

        self.optimizer.zero_grad()
        loss.backward()

        # trainable_params = []
        # for m in self.get_trainable_models():
        #     trainable_params.extend(m.parameters())
        # torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=self.max_grad_norm)
        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        self.global_step += 1
        return {"loss": loss.item()}

    def validate_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            token_ids = batch["token_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            label_spans = batch["label_spans"].to(self.device)
            B, L = token_ids.shape

            pred_spans = self.model.predict(token_ids, attention_mask)
            # this is big costil
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
            B, L = token_ids.shape

            token_embeddings = self.model.encode_tokens(token_ids, attention_mask)
            x_0 = self.model.label_mapper.forward(label_spans, L).reshape(B, 6, L)
            t = torch.randint(
                0,
                self.model.scheduler.num_steps,
                size=(B,),
                device=self.device,
                dtype=torch.long,
            )
            x_t = self.model.noise(x_0, t)
            x0_pred = self.model.denoiser(
                x_t=x_t,
                t=t,
                token_embeddings=token_embeddings,
                attn_mask=attention_mask.bool(),
            )
            mask = attention_mask.unsqueeze(1).expand_as(x_0)
            loss = (self.criterion(x0_pred, x_0) * mask).sum() / mask.sum().clamp(min=1)
            return {"loss": loss.item()}

    def get_checkpoint_state_dict(self) -> Dict[str, torch.Tensor]:
        sd = self.model.state_dict()
        return {k: v for k, v in sd.items() if not k.startswith("encoder.")}

    def load_checkpoint_state_dict(self, checkpoint: Dict[str, torch.Tensor]):
        to_load = {k: v for k, v in checkpoint.items() if not k.startswith("encoder.")}
        self.model.load_state_dict(to_load, strict=False)


class SpanDiffusionTrainerConfig(BaseTrainerConfig):
    type: Literal["span_trainer"] = "span_trainer"

    def create(self, model: SpanDiffusionModel) -> SpanDiffusionTrainer:
        return SpanDiffusionTrainer(
            model=model,
            device=self.device,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            max_grad_norm=self.max_grad_norm,
        )

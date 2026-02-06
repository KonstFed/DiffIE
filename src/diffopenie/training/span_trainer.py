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
        loss_type: Literal["mse", "cross_entropy"] = "mse",
    ):
        self.model = model.to(device)
        super().__init__(
            device=device,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
        )
        self.loss_type = loss_type
        self.criterion = (
            nn.MSELoss(reduction="none")
            if loss_type == "mse"
            else nn.CrossEntropyLoss(reduction="mean")
        )

    def get_trainable_models(self) -> List[nn.Module]:
        return [self.model.denoiser]

    def get_eval_models(self) -> List[nn.Module]:
        return [self.model.scheduler, self.model.encoder]

    def _compute_loss(
        self,
        x0_pred: torch.Tensor,
        x_0: torch.Tensor,
        label_spans: torch.Tensor,
        attention_mask: torch.Tensor,
        seq_len: torch.Tensor,
    ) -> torch.Tensor:
        """Compute loss (MSE or Cross Entropy) on valid positions.

        x0_pred, x_0: [B, 6, L-1]; label_spans: [B, 6]; seq_len: [B].
        """
        mask = attention_mask[:, :-1].unsqueeze(1).expand_as(x_0)
        if self.loss_type == "mse":
            loss_per_element = self.criterion(x0_pred, x_0)
            return (loss_per_element * mask).sum() / mask.sum().clamp(min=1)
        else:
            B, _, Lm1 = x0_pred.shape
            valid_logits = (
                torch.arange(Lm1, device=x0_pred.device).unsqueeze(0)
                < (seq_len - 1).unsqueeze(1)
            )
            logits_masked = x0_pred.masked_fill(
                ~valid_logits.unsqueeze(1).expand_as(x0_pred), float("-inf")
            )
            max_idx = (seq_len - 2).unsqueeze(1).expand(-1, 6).clamp(min=0)
            labels_clamped = label_spans.clamp(max=max_idx).reshape(B * 6)
            logits_flat = logits_masked.reshape(B * 6, Lm1)
            return self.criterion(logits_flat, labels_clamped)

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        token_ids = batch["token_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        label_spans = batch["label_spans"].to(self.device)
        seq_len = batch["seq_len"].to(self.device)
        B, L = token_ids.shape

        token_embeddings = batch["token_embeddings"].to(self.device)
        x_0 = self.model.label_mapper.forward(label_spans, seq_len).reshape(
            B, 6, -1
        )
        assert x_0.shape[2] == L - 1, (
            f"x_0 logit dim {x_0.shape[2]} != L-1 {L - 1}; "
            "collator should pad to max length in batch so max(seq_len)=L."
        )
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
        loss = self._compute_loss(
            x0_pred, x_0, label_spans, attention_mask, seq_len
        )

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
            x_0 = self.model.label_mapper.forward(label_spans, seq_len).reshape(
                B, 6, -1
            )
            assert x_0.shape[2] == L - 1, (
                f"x_0 logit dim {x_0.shape[2]} != L-1 {L - 1}; "
                "collator should pad to max length in batch so max(seq_len)=L."
            )
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
            loss = self._compute_loss(
                x0_pred, x_0, label_spans, attention_mask, seq_len
            )
            return {"loss": loss.item()}

    def get_checkpoint_state_dict(self) -> Dict[str, torch.Tensor]:
        sd = self.model.state_dict()
        return {k: v for k, v in sd.items() if not k.startswith("encoder.")}

    def load_checkpoint_state_dict(self, checkpoint: Dict[str, torch.Tensor]):
        to_load = {k: v for k, v in checkpoint.items() if not k.startswith("encoder.")}
        self.model.load_state_dict(to_load, strict=False)


class SpanDiffusionTrainerConfig(BaseTrainerConfig):
    type: Literal["span_trainer"] = "span_trainer"
    loss_type: Literal["mse", "cross_entropy"] = "mse"

    def create(self, model: SpanDiffusionModel) -> SpanDiffusionTrainer:
        return SpanDiffusionTrainer(
            model=model,
            device=self.device,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            max_grad_norm=self.max_grad_norm,
            loss_type=self.loss_type,
        )

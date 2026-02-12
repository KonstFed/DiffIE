"""Trainer for DetIE: single-pass OpenIE (single classification or optional bipartite matching)."""

from typing import Dict, List, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffopenie.models.detie import DetIEModel, DetIELoss
from diffopenie.models.span.span_model import spans_to_token_labels
from diffopenie.training.base_trainer import BaseTrainer, BaseTrainerConfig


def _build_gold_labels_for_detie(
    label_spans: torch.LongTensor,
    attention_mask: torch.Tensor,
    num_slots: int,
    device: torch.device,
) -> torch.LongTensor:
    """
    Convert [B, 6] label_spans to [B, N, L] for DetIE bipartite-matching loss.
    First slot = real triplet (spans -> token labels), rest = 0 (no triplet).
    """
    B, L = label_spans.shape[0], attention_mask.shape[1]
    gold_single = spans_to_token_labels(label_spans, L)  # [B, L]
    gold_single = gold_single.to(device)
    gold_labels = gold_single.unsqueeze(1)  # [B, 1, L]
    pad_slots = torch.zeros(B, num_slots - 1, L, dtype=torch.long, device=device)
    gold_labels = torch.cat([gold_labels, pad_slots], dim=1)  # [B, N, L]
    return gold_labels


class DetIETrainer(BaseTrainer):
    """
    Trainer for DetIE: single classification on slot 0 (no DETR/Hungarian),
    or optional bipartite matching loss. Validation uses slot-0 predictions and CaRB metrics.
    """

    def __init__(
        self,
        model: DetIEModel,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        use_bipartite_matching: bool = False,
        matching: str = "dice",
        no_object_weight: float = 0.1,
    ):
        self.model = model.to(device)
        self.use_bipartite_matching = use_bipartite_matching
        super().__init__(
            device=device,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
        )
        self.bipartite_criterion = (
            DetIELoss(
                num_classes=model.num_classes,
                matching=matching,
                no_object_weight=no_object_weight,
            )
            if use_bipartite_matching
            else None
        )

    def get_trainable_models(self) -> List[nn.Module]:
        return [self.model]

    def get_eval_models(self) -> List[nn.Module]:
        return [self.model]

    def _loss_single_classification(
        self,
        pred_logits: torch.Tensor,
        gold_labels: torch.LongTensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Cross-entropy on slot 0 only; mask padding."""
        B, N, L, C = pred_logits.shape
        logits = pred_logits[:, 0].reshape(-1, C)  # [B*L, C]
        gold_flat = gold_labels.reshape(-1).clone()  # [B*L]
        mask = attention_mask.reshape(-1).bool()  # [B*L]
        gold_flat[~mask] = -100
        return F.cross_entropy(logits, gold_flat, ignore_index=-100)

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        token_ids = batch["token_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        label_spans = batch["label_spans"].to(self.device)
        B, L = token_ids.shape
        dev = torch.device(self.device)

        pred_logits = self.model(token_ids, attention_mask)

        if self.use_bipartite_matching:
            num_slots = self.model.num_slots
            gold_labels = _build_gold_labels_for_detie(
                label_spans, attention_mask, num_slots, dev
            )
            loss, aux = self.bipartite_criterion(pred_logits, gold_labels, attention_mask)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.max_grad_norm
            )
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self.global_step += 1
            return {
                "loss": loss.item(),
                "num_matched": aux["num_matched"],
                "num_no_obj": aux["num_no_obj"],
            }
        else:
            gold_labels = spans_to_token_labels(label_spans, L).to(dev)
            loss = self._loss_single_classification(
                pred_logits, gold_labels, attention_mask
            )
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.max_grad_norm
            )
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

            pred_logits = self.model(token_ids, attention_mask)
            pred_labels = pred_logits[:, 0].argmax(dim=-1).long()
            pred_labels = pred_labels * attention_mask.long()

            gold_labels = spans_to_token_labels(label_spans, L)
            return {
                "predictions": pred_labels,
                "labels": gold_labels.to(self.device),
                "attention_mask": attention_mask,
            }

    def validate_loss_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        with torch.no_grad():
            token_ids = batch["token_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            label_spans = batch["label_spans"].to(self.device)
            B, L = token_ids.shape
            dev = torch.device(self.device)

            pred_logits = self.model(token_ids, attention_mask)
            if self.use_bipartite_matching:
                num_slots = self.model.num_slots
                gold_labels = _build_gold_labels_for_detie(
                    label_spans, attention_mask, num_slots, dev
                )
                loss, _ = self.bipartite_criterion(
                    pred_logits, gold_labels, attention_mask
                )
            else:
                gold_labels = spans_to_token_labels(label_spans, L).to(dev)
                loss = self._loss_single_classification(
                    pred_logits, gold_labels, attention_mask
                )
            return {"loss": loss.item()}

    def get_checkpoint_state_dict(self) -> Dict[str, torch.Tensor]:
        return self.model.state_dict()

    def load_checkpoint_state_dict(self, checkpoint: Dict[str, torch.Tensor]):
        self.model.load_state_dict(checkpoint, strict=False)


class DetIETrainerConfig(BaseTrainerConfig):
    type: Literal["detie_trainer"] = "detie_trainer"
    use_bipartite_matching: bool = False
    matching: str = "dice"
    no_object_weight: float = 0.1

    def create(self, model: DetIEModel) -> DetIETrainer:
        return DetIETrainer(
            model=model,
            device=self.device,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            max_grad_norm=self.max_grad_norm,
            use_bipartite_matching=self.use_bipartite_matching,
            matching=self.matching,
            no_object_weight=self.no_object_weight,
        )

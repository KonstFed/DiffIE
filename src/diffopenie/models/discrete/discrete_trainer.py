from typing import Dict, Literal

import torch

from diffopenie.models.discrete.discrete_model import DiscreteModel
from diffopenie.training.base_trainer import BaseTrainer, BaseTrainerConfig


class DiscreteTrainer(BaseTrainer):
    def __init__(
        self,
        model: DiscreteModel,
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
        self._ignore_index = -100

    def get_trainable_models(self):
        return [self.model]

    def get_eval_models(self):
        return [self.model]

    def _forward_loss(self, batch: Dict[str, torch.Tensor]):
        token_ids = batch["token_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["label_indices"].to(self.device)
        B, L = token_ids.shape
        token_emb = self.model.encode_tokens(token_ids, attention_mask)
        t = self.model.scheduler.sample_t(B)
        x_t = self.model.noise(labels, t)
        logits = self.model.denoiser(x_t, t, token_emb, attention_mask)
        target = labels.clone()

        # ignore paddings
        target[attention_mask == 0] = self._ignore_index
        if self.model.scheduler.kernel == "mask_absorbing":
            # if MASK tokens are used we should ignore all prediction from non-MASK tokens
            mask_state_id = self.model.scheduler.mask_state_id
            target[x_t != mask_state_id] = self._ignore_index

        return torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target.view(-1),
            ignore_index=self._ignore_index,
        )

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        loss = self._forward_loss(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.optimizer.zero_grad()
        return {"loss": loss.item()}

    def validate_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            token_ids = batch["token_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["label_indices"].to(self.device)
            B = token_ids.shape[0]
            token_emb = self.model.encode_tokens(token_ids, attention_mask)
            pred = self.model.generate(
                batch_size=B,
                token_embeddings=token_emb,
                attention_mask=attention_mask,
            )
            return {
                "predictions": pred,
                "labels": labels,
                "attention_mask": attention_mask,
            }

    def validate_loss_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        with torch.no_grad():
            loss = self._forward_loss(batch)
        return {"loss": loss.item()}

    def get_checkpoint_state_dict(self) -> Dict[str, torch.Tensor]:
        return self.model.state_dict()

    def load_checkpoint_state_dict(self, checkpoint: Dict[str, torch.Tensor]):
        self.model.load_state_dict(checkpoint, strict=False)


class DiscreteTrainerConfig(BaseTrainerConfig):
    """Configuration for DiscreteTrainer."""

    type: Literal["discrete_trainer"] = "discrete_trainer"

    def create(self, model: DiscreteModel) -> DiscreteTrainer:
        return DiscreteTrainer(
            model=model,
            device=self.device,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            max_grad_norm=self.max_grad_norm,
        )

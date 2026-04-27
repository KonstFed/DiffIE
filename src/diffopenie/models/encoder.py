"""BERT encoder wrapper for generating token embeddings."""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from pydantic import BaseModel, ConfigDict


class BERTEncoder(nn.Module):
    """
    Thin wrapper around a HuggingFace encoder model.

    Special tokens (CLS/SEP) are expected to already be in `input_ids` —
    the dataset/inference tokenizer call should be made with
    `add_special_tokens=True`. Labels at CLS/SEP positions naturally fall on
    word_ids() == None, which downstream span extraction already skips.

    Args:
        model_name: HuggingFace model name.
        freeze: Freeze the entire encoder.
        num_frozen_layers: When >0 and `freeze=False`, freezes word/position
            embeddings and the first K transformer layers. Cuts optimizer
            state and gradient memory; backward pass also skips those layers.
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        freeze: bool = False,
        num_frozen_layers: int = 0,
    ):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.bert_dim = self.model.config.hidden_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False
        elif num_frozen_layers > 0:
            self._freeze_bottom_layers(num_frozen_layers)

    def _freeze_bottom_layers(self, k: int) -> None:
        for p in self.model.embeddings.parameters():
            p.requires_grad = False
        layers = self.model.encoder.layer
        k = min(k, len(layers))
        for i in range(k):
            for p in layers[i].parameters():
                p.requires_grad = False

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state


class BERTEncoderConfig(BaseModel):
    """Configuration model for BERTEncoder."""

    model_config = ConfigDict(extra="forbid")
    model_name: str = "bert-base-uncased"
    freeze: bool = False
    num_frozen_layers: int = 0

    def create(self) -> BERTEncoder:
        return BERTEncoder(
            model_name=self.model_name,
            freeze=self.freeze,
            num_frozen_layers=self.num_frozen_layers,
        )

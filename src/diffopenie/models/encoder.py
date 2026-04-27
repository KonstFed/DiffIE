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
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        freeze: bool = False,
    ):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.bert_dim = self.model.config.hidden_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

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

    def create(self) -> BERTEncoder:
        return BERTEncoder(
            model_name=self.model_name,
            freeze=self.freeze,
        )

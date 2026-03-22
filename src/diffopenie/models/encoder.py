"""BERT encoder wrapper for generating token embeddings."""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from pydantic import BaseModel, ConfigDict


class BERTEncoder(nn.Module):
    """
    Wrapper for BERT (or other transformer) encoder to extract token embeddings.

    Adds special tokens (CLS, SEP) to match BERT's expected input format,
    even if the dataset doesn't include them. The special tokens are added
    during encoding but excluded from the output to maintain length consistency
    with labels.

    Args:
        model_name: HuggingFace model name (e.g., "bert-base-uncased")
        freeze: Whether to freeze the encoder weights
        add_special_tokens: Whether to add CLS and SEP tokens (default: True)
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        freeze: bool = True,
        add_special_tokens: bool = True,
    ):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.bert_dim = self.model.config.hidden_size
        self.add_special_tokens = add_special_tokens

        # Get tokenizer to access special token IDs
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(
        self,
        input_ids: torch.LongTensor,  # [B, L]
        attention_mask: torch.Tensor | None = None,  # [B, L]
    ) -> torch.Tensor:
        """
        Extract token embeddings from BERT.

        If add_special_tokens=True, prepends CLS and appends SEP tokens to the input,
        but returns embeddings only for the original tokens (excluding special tokens)
        to maintain length consistency with labels.

        Returns:
            token_embeddings: [B, L, bert_dim] (same length as input, excluding special tokens)
        """
        if self.add_special_tokens:
            B, L = input_ids.shape
            device = input_ids.device

            if attention_mask is not None:
                lengths = attention_mask.sum(dim=1).long()  # [B] real token count
            else:
                lengths = torch.full((B,), L, dtype=torch.long, device=device)

            # [B, L+2]: CLS <real tokens> SEP <PAD ...>
            new_ids = torch.full(
                (B, L + 2),
                self.tokenizer.pad_token_id,
                dtype=torch.long,
                device=device,
            )
            new_mask = torch.zeros(
                (B, L + 2),
                dtype=torch.long,
                device=device,
            )

            new_ids[:, 0] = self.cls_token_id
            new_mask[:, 0] = 1

            # Copy real tokens and place SEP right after each sample
            for i in range(B):
                sl = lengths[i]
                new_ids[i, 1 : 1 + sl] = input_ids[i, :sl]
                new_mask[i, 1 : 1 + sl] = 1
                new_ids[i, 1 + sl] = self.sep_token_id
                new_mask[i, 1 + sl] = 1

            outputs = self.model(input_ids=new_ids, attention_mask=new_mask)
            hidden = outputs.last_hidden_state  # [B, L+2, bert_dim]

            # Extract only real-token embeddings (skip CLS and SEP)
            token_embeddings = torch.zeros(
                B,
                L,
                self.bert_dim,
                dtype=hidden.dtype,
                device=device,
            )
            for i in range(B):
                sl = lengths[i]
                token_embeddings[i, :sl] = hidden[i, 1 : 1 + sl]
        else:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            token_embeddings = outputs.last_hidden_state

        return token_embeddings


class BERTEncoderConfig(BaseModel):
    """
    Configuration model for BERTEncoder.
    Acts as a factory for creating BERTEncoder instances.
    """

    model_config = ConfigDict(extra="forbid")
    model_name: str = "bert-base-uncased"
    freeze: bool = True
    add_special_tokens: bool = True

    def create(self) -> BERTEncoder:
        """
        Factory method to create a BERTEncoder instance.

        Returns:
            Instance of BERTEncoder

        Example:
            config = BERTEncoderConfig(model_name="bert-base-uncased", freeze=True)
            encoder = config.create()
        """
        return BERTEncoder(
            model_name=self.model_name,
            freeze=self.freeze,
            add_special_tokens=self.add_special_tokens,
        )

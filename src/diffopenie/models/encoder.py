"""BERT encoder wrapper for generating token embeddings."""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from pydantic import BaseModel


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
            # Add CLS token at the beginning and SEP token at the end
            B, L = input_ids.shape
            device = input_ids.device

            # Create new input_ids with special tokens
            cls_tokens = torch.full(
                (B, 1), self.cls_token_id, dtype=torch.long, device=device
            )
            sep_tokens = torch.full(
                (B, 1), self.sep_token_id, dtype=torch.long, device=device
            )
            input_ids_with_special = torch.cat(
                [cls_tokens, input_ids, sep_tokens], dim=1
            )  # [B, L+2]

            # Update attention mask
            if attention_mask is not None:
                cls_mask = torch.ones((B, 1), dtype=attention_mask.dtype, device=device)
                sep_mask = torch.ones((B, 1), dtype=attention_mask.dtype, device=device)
                attention_mask_with_special = torch.cat(
                    [cls_mask, attention_mask, sep_mask], dim=1
                )  # [B, L+2]
            else:
                attention_mask_with_special = None
        else:
            input_ids_with_special = input_ids
            attention_mask_with_special = attention_mask

        outputs = self.model(
            input_ids=input_ids_with_special,
            attention_mask=attention_mask_with_special,
        )
        # Use last hidden state (token embeddings)
        token_embeddings = (
            outputs.last_hidden_state
        )  # [B, L+2, bert_dim] or [B, L, bert_dim]

        # If we added special tokens, remove them from the output to maintain length consistency
        if self.add_special_tokens:
            # Remove CLS (first) and SEP (last) token embeddings
            token_embeddings = token_embeddings[:, 1:-1, :]  # [B, L, bert_dim]

        return token_embeddings


class BERTEncoderConfig(BaseModel):
    """
    Configuration model for BERTEncoder.
    Acts as a factory for creating BERTEncoder instances.
    """

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

"""Data collator for batching with padding."""

import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from typing import Any, Dict, List


# ContinuousSpanMapper.forward expects [B, 6]: (S_l, S_r, O_l, O_r, P_l, P_r).
SPAN_LABEL_ORDER = ("S_l", "S_r", "O_l", "O_r", "P_l", "P_r")


def _span_to_indices(span: tuple) -> tuple[int, int]:
    """(start, end) or (None, None) -> (int, int); use -1 for missing."""
    left, right = span
    return (
        -1 if left is None else left,
        -1 if right is None else right,
    )


class SequenceCollator:
    """
    Collator for batching sequence training data.

    Handles:
    - Padding context token_ids and context attention masks
    - Padding tag tensors and tag attention masks
    """

    def __init__(
        self,
        pad_token_id: int = 0,
        pad_tag_value: int = 0,
    ):
        """
        Args:
            pad_token_id: Token ID for padding
            pad_tag_value: Value used to pad tag tensors. This value is masked
                out by tag_attention_mask and is not part of the diffusion state.
        """
        self.pad_token_id = pad_token_id
        self.pad_tag_value = pad_tag_value

    @staticmethod
    def _as_long_tensor(value: Any) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.clone().long()
        return torch.tensor(value, dtype=torch.long)

    def _pad_sequences(
        self,
        values: List[Any],
        pad_value: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pad a list of 1D sequences and return (padded, attention_mask)."""
        tensors = [self._as_long_tensor(value) for value in values]
        lengths = [t.size(0) for t in tensors]
        padded = pad_sequence(
            tensors,
            batch_first=True,
            padding_value=pad_value,
        )
        mask = torch.zeros(len(tensors), padded.size(1), dtype=torch.long)
        for i, length in enumerate(lengths):
            mask[i, :length] = 1
        return padded, mask

    def _pad_tokens(
        self, batch: List[Dict[str, Any]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Backward-compatible wrapper for padding context token ids."""
        values = [
            item.get("context_token_ids", item.get("token_ids")) for item in batch
        ]
        if any(value is None for value in values):
            raise ValueError("Batch item missing context_token_ids/token_ids")
        return self._pad_sequences(values, pad_value=self.pad_token_id)

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of examples.

        Expected batch items:
        - context_token_ids or token_ids: List[int]
        - tag_ids or labels: torch.Tensor of shape [L] with tag indices
          (0=B, 1=S, 2=R, 3=O)
        """
        context_values = [
            item.get("context_token_ids", item.get("token_ids")) for item in batch
        ]
        tag_values = [item.get("tag_ids", item.get("labels")) for item in batch]

        if any(value is None for value in context_values):
            raise ValueError("Batch item missing context_token_ids/token_ids")
        if any(value is None for value in tag_values):
            raise ValueError("Batch item missing tag_ids/labels")

        context_token_ids, context_attention_mask = self._pad_sequences(
            context_values,
            pad_value=self.pad_token_id,
        )
        tag_ids, tag_attention_mask = self._pad_sequences(
            tag_values,
            pad_value=self.pad_tag_value,
        )

        return {
            "context_token_ids": context_token_ids,
            "context_attention_mask": context_attention_mask,
            "tag_ids": tag_ids,
            "tag_attention_mask": tag_attention_mask,
            # Legacy aliases
            "token_ids": context_token_ids,
            "attention_mask": context_attention_mask,
            "label_indices": tag_ids,
            "state_mask": tag_attention_mask,
        }


class SequenceGroupedCollator(SequenceCollator):
    """Collator for grouped datasets.

    GroupedImojieDataset / GroupedSequenceLSOEIDataset return list[dict] from
    __getitem__ — one entry per triplet variant of a sentence — so all variants
    of the same sentence land in the same batch. Each entry already has its
    context concatenated into `context_token_ids`; the tag sequence stays tied
    to the original sentence length. We flatten and delegate to the parent
    collator.
    """

    def __call__(self, batch: List[Any]) -> Dict[str, Any]:
        if batch and isinstance(batch[0], list):
            batch = [item for sublist in batch for item in sublist]
        return super().__call__(batch)


def _pad_embeddings(
    embeddings_list: List[torch.Tensor | np.ndarray],
    max_len: int,
    embed_dim: int,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Pad variable-length embeddings [L_i, D] to [B, L, D] with zeros."""
    B = len(embeddings_list)
    padded = torch.zeros(B, max_len, embed_dim, dtype=dtype)
    for i, emb in enumerate(embeddings_list):
        if isinstance(emb, np.ndarray):
            emb = torch.from_numpy(emb).to(dtype=dtype)
        else:
            emb = emb.to(dtype=dtype)
        L_i = emb.size(0)
        padded[i, :L_i] = emb
    if device is not None:
        padded = padded.to(device)
    return padded


class SpanCollator(SequenceCollator):
    """
    Collator for batching span training data (subject/object/predicate spans).

    When batch items contain "token_embeddings" (precomputed), pads them to
    [B, L, D] and returns "token_embeddings" in the batch.

    Outputs label_spans [B, 6] in ContinuousSpanMapper format:
    (S_l, S_r, O_l, O_r, P_l, P_r). Uses -1 for missing span bounds.
    """

    def __init__(self, pad_token_id: int = 0, **kwargs: Any):
        super().__init__(pad_token_id=pad_token_id, pad_tag_value=0, **kwargs)

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of span examples.

        Expected batch items (from SpanLSOIEDataset):
        - token_ids: List[int]
        - subject_span: (int, int) or (None, None)
        - object_span: (int, int) or (None, None)
        - predicate_span: (int, int) or (None, None)
        - token_embeddings: optional [L_i, D] array/tensor (precomputed)

        Returns token_ids, attention_mask, label_spans [B, 6], seq_len [B],
        and token_embeddings [B, L, D] when present in items.
        """
        token_ids, attention_mask = self._pad_tokens(batch)

        # [B, 6] in label_mapper order: S_l, S_r, O_l, O_r, P_l, P_r
        label_spans_list = []
        for item in batch:
            s_l, s_r = _span_to_indices(item["subject_span"])
            o_l, o_r = _span_to_indices(item["object_span"])
            p_l, p_r = _span_to_indices(item["predicate_span"])
            label_spans_list.append([s_l, s_r, o_l, o_r, p_l, p_r])

        label_spans = torch.tensor(label_spans_list, dtype=torch.long)  # [B, 6]
        seq_len = attention_mask.sum(dim=1).clamp(min=2).long()  # [B], for label_mapper

        out: Dict[str, torch.Tensor] = {
            "context_token_ids": token_ids,
            "context_attention_mask": attention_mask,
            "token_ids": token_ids,
            "attention_mask": attention_mask,
            "label_spans": label_spans,
            "seq_len": seq_len,
        }

        # Pad precomputed token_embeddings when present (train with precomputed embs)
        has_embs = (
            batch
            and "token_embeddings" in batch[0]
            and batch[0]["token_embeddings"] is not None
        )
        if has_embs:
            _, L = token_ids.shape
            emb_list = [item["token_embeddings"] for item in batch]
            first = emb_list[0]
            if isinstance(first, np.ndarray):
                embed_dim = first.shape[1]
            else:
                embed_dim = first.shape[1]
            token_embeddings = _pad_embeddings(
                emb_list, L, embed_dim, device=None, dtype=torch.float32
            )
            out["token_embeddings"] = token_embeddings

        return out

"""Data collator for batching with padding."""

import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from typing import Dict, List, Any


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
    - Padding token_ids and attention masks
    - Padding label tensors
    - Creating attention masks
    """

    def __init__(
        self,
        pad_token_id: int = 0,
        pad_label_idx: int = 0,
    ):
        """
        Args:
            pad_token_id: Token ID for padding
            pad_label_idx: Label index for padding (typically 0 for O/padding)
        """
        self.pad_token_id = pad_token_id
        self.pad_label_idx = pad_label_idx

    def _pad_tokens(
        self, batch: List[Dict[str, Any]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pad token_ids and build attention_mask. Returns (token_ids, mask)."""
        token_ids_list = [
            torch.tensor(item["token_ids"], dtype=torch.long) for item in batch
        ]
        token_ids = pad_sequence(
            token_ids_list,
            batch_first=True,
            padding_value=self.pad_token_id,
        )  # [B, L]
        attention_mask = (token_ids != self.pad_token_id).long()  # [B, L]
        return token_ids, attention_mask

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of examples.

        Expected batch items:
        - token_ids: List[int]
        - labels: torch.Tensor of shape [L] with label indices
          (0=O, 1=subject, 2=object, 3=predicate)
        """
        token_ids, attention_mask = self._pad_tokens(batch)
        labels_list = [item.get("labels") for item in batch]
        max_len = token_ids.size(1)
        label_indices_list = []

        for i, labels in enumerate(labels_list):
            seq_len = len(batch[i]["token_ids"])
            if labels is None:
                label_indices = torch.zeros(seq_len, dtype=torch.long)
            elif isinstance(labels, torch.Tensor):
                label_indices = labels.clone()
            else:
                label_indices = torch.tensor(labels, dtype=torch.long)

            if len(label_indices) < max_len:
                padding = torch.full(
                    (max_len - len(label_indices),),
                    self.pad_label_idx,
                    dtype=torch.long,
                )
                label_indices = torch.cat([label_indices, padding])
            elif len(label_indices) > max_len:
                label_indices = label_indices[:max_len]

            label_indices_list.append(label_indices)

        label_indices = torch.stack(label_indices_list)  # [B, L]

        return {
            "token_ids": token_ids,
            "attention_mask": attention_mask,
            "label_indices": label_indices,
        }


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
        super().__init__(pad_token_id=pad_token_id, pad_label_idx=0, **kwargs)

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

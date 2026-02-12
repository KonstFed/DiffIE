"""DetIE: single-pass OpenIE with bipartite matching (DETR-style).

Transformer encoder-only + N slot queries; each slot predicts a sequence of
labels (O, Subject, Object, Predicate). Bipartite matching assigns predictions
to gold triplets.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from typing import Optional

from diffopenie.models.base_model import BaseTripletModel
from diffopenie.models.encoder import BERTEncoder


class SlotDecoder(nn.Module):
    """Transformer decoder: slot queries attend to encoder output."""

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        n_layers: int = 2,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=n_layers)

    def forward(
        self,
        slot_queries: torch.Tensor,
        encoder_output: torch.Tensor,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        slot_queries: [B, N, d_model]
        encoder_output: [B, L, d_model]
        memory_key_padding_mask: [B, L] True = ignore

        Returns: [B, N, d_model]
        """
        # TransformerDecoder expects memory_key_padding_mask True = ignore
        out = self.decoder(
            slot_queries,
            encoder_output,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return out


def _slot_logits_to_single_span_labels(
    pred_logits: torch.Tensor,
    attention_mask: torch.Tensor,
    slot_idx: int = 0,
) -> torch.LongTensor:
    """
    From [B, N, L, C] take slot_idx and argmax to get [B, L] label indices.
    Used for validation when we take the first slot as the primary triplet.
    """
    B, N, L, C = pred_logits.shape
    logits = pred_logits[:, slot_idx]  # [B, L, C]
    pred = logits.argmax(dim=-1).long()  # [B, L]
    pred = pred * attention_mask.long() + (1 - attention_mask.long()) * 0
    return pred


def _slot_logits_to_spans(pred_logits: torch.Tensor, seq_len: int) -> torch.LongTensor:
    """
    Convert [B, N, L, C] to [B, 6] by taking slot 0 and extracting
    (s_l, s_r, o_l, o_r, p_l, p_r) from argmax positions.
    Label: 0=O, 1=subject, 2=object, 3=predicate.
    """
    B = pred_logits.shape[0]
    device = pred_logits.device
    out = torch.zeros(B, 6, dtype=torch.long, device=device)
    logits = pred_logits[:, 0]  # [B, L, C]
    L = logits.shape[1]
    pred_labels = logits.argmax(dim=-1)  # [B, L]

    for b in range(B):
        for c, idx in [(1, 0), (2, 2), (3, 4)]:  # subject -> 0,1; object -> 2,3; predicate -> 4,5
            pos = (pred_labels[b] == c).nonzero(as_tuple=True)[0]
            if pos.numel() > 0:
                out[b, idx] = pos.min().item()
                out[b, idx + 1] = pos.max().item()
            else:
                out[b, idx] = 0
                out[b, idx + 1] = 0

    out[:, 0:2] = out[:, 0:2].clamp(0, seq_len - 1)
    out[:, 2:4] = out[:, 2:4].clamp(0, seq_len - 1)
    out[:, 4:6] = out[:, 4:6].clamp(0, seq_len - 1)
    return out


class DetIEModel(nn.Module, BaseTripletModel):
    """
    DetIE: encoder + N slot queries; each slot predicts [L, 4] label logits.
    """

    def __init__(
        self,
        encoder: BERTEncoder,
        num_slots: int = 20,
        d_model: int = 512,
        n_heads: int = 8,
        n_decoder_layers: int = 2,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        num_classes: int = 4,
    ):
        super().__init__()
        self.encoder = encoder
        self.num_slots = num_slots
        self.num_classes = num_classes
        bert_dim = encoder.bert_dim
        self.proj = nn.Linear(bert_dim, d_model) if bert_dim != d_model else nn.Identity()
        self.slot_embeddings = nn.Parameter(torch.randn(1, num_slots, d_model) * 0.02)
        self.decoder = SlotDecoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_decoder_layers,
            d_ff=d_ff,
            dropout=dropout,
        )
        self.head = nn.Linear(d_model * 2, num_classes)
        self.d_model = d_model

    def encode_tokens(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """[B, L] -> [B, L, d_model]."""
        h = self.encoder(input_ids, attention_mask)
        return self.proj(h)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns pred_logits [B, N, L, C].
        """
        B, L = input_ids.shape
        encoder_out = self.encode_tokens(input_ids, attention_mask)  # [B, L, d_model]
        slots = self.slot_embeddings.expand(B, -1, -1)  # [B, N, d_model]
        # True = padding (ignore)
        memory_key_padding_mask = (1 - attention_mask).bool()  # [B, L]
        slot_out = self.decoder(slots, encoder_out, memory_key_padding_mask=memory_key_padding_mask)  # [B, N, d_model]
        # Per-position logits: concat slot vector with encoder position vector
        slot_exp = slot_out.unsqueeze(2).expand(-1, -1, L, -1)   # [B, N, L, d_model]
        enc_exp = encoder_out.unsqueeze(1).expand(-1, self.num_slots, -1, -1)  # [B, N, L, d_model]
        logits = self.head(torch.cat([slot_exp, enc_exp], dim=-1))  # [B, N, L, C]
        return logits

    def forward_from_embeddings(
        self,
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Same as forward but from precomputed [B, L, bert_dim] embeddings."""
        B, L, _ = token_embeddings.shape
        encoder_out = self.proj(token_embeddings)
        slots = self.slot_embeddings.expand(B, -1, -1)
        memory_key_padding_mask = (1 - attention_mask).bool()
        slot_out = self.decoder(slots, encoder_out, memory_key_padding_mask=memory_key_padding_mask)
        slot_exp = slot_out.unsqueeze(2).expand(-1, -1, L, -1)
        enc_exp = encoder_out.unsqueeze(1).expand(-1, self.num_slots, -1, -1)
        logits = self.head(torch.cat([slot_exp, enc_exp], dim=-1))
        return logits

    @torch.no_grad()
    def predict(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
    ) -> torch.LongTensor:
        """Predict [B, L] label indices (primary slot)."""
        logits = self.forward(input_ids, attention_mask)
        return _slot_logits_to_single_span_labels(logits, attention_mask, slot_idx=0)

    @torch.no_grad()
    def predict_spans(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
    ) -> torch.LongTensor:
        """Predict [B, 6] span indices (s_l, s_r, o_l, o_r, p_l, p_r) from slot 0."""
        logits = self.forward(input_ids, attention_mask)
        L = attention_mask.sum(dim=1).max().item()
        return _slot_logits_to_spans(logits, int(L))

    @torch.no_grad()
    def get_triplets(
        self, words: list[list[str]]
    ) -> list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]]:
        """Get triplets as word spans for a batch of word lists."""
        if not words:
            return []
        device = next(self.parameters()).device
        encodings = self.encoder.tokenizer(
            words,
            is_split_into_words=True,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        )
        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)
        pred_spans = self.predict_spans(input_ids, attention_mask).cpu()
        num_words = [len(w) for w in words]
        results = []
        for i in range(len(words)):
            word_ids_list = encodings.word_ids(batch_index=i)
            L = len(word_ids_list)
            nw = num_words[i]

            def token_to_word(tok_idx: int, default: int) -> int:
                idx = max(0, min(int(tok_idx), L - 1))
                w = word_ids_list[idx]
                return w if w is not None else default

            s_l = token_to_word(pred_spans[i, 0].item(), 0)
            s_r = token_to_word(pred_spans[i, 1].item(), 0)
            o_l = token_to_word(pred_spans[i, 2].item(), 0)
            o_r = token_to_word(pred_spans[i, 3].item(), 0)
            p_l = token_to_word(pred_spans[i, 4].item(), 0)
            p_r = token_to_word(pred_spans[i, 5].item(), 0)
            s_l, s_r = max(0, min(s_l, nw - 1)), max(0, min(s_r, nw - 1))
            o_l, o_r = max(0, min(o_l, nw - 1)), max(0, min(o_r, nw - 1))
            p_l, p_r = max(0, min(p_l, nw - 1)), max(0, min(p_r, nw - 1))
            if s_l > s_r:
                s_r = s_l
            if o_l > o_r:
                o_r = o_l
            if p_l > p_r:
                p_r = p_l
            results.append(((s_l, s_r), (o_l, o_r), (p_l, p_r)))
        return results

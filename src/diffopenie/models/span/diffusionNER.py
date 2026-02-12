# acutally IPED inspired. TODO: sort it out

import torch
import torch.nn as nn
from diffopenie.diffusion.scheduler import BaseDenoiser
from diffopenie.models.span.label_mapper import FloatIndexMapper


class SpanQueryBlock(nn.Module):
    """Use mean of span"""
    def __init__(self, token_dim: int, span_dim: int, dropout: float = 0.1):
        super().__init__()
        self.lin = nn.Sequential(
            nn.Linear(token_dim, span_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(span_dim, span_dim),
        )

    def forward(
        self,
        left: torch.LongTensor,
        right: torch.LongTensor,
        token_embeddings: torch.Tensor,
    ) -> torch.FloatTensor:
        """Get span representation from token embeddings

        Args:
            left (torch.LongTensor): [B]
            right (torch.LongTensor): [B]
            token_embeddings (torch.Tensor): [B, L, D] padded token embeddings

        Returns:
            torch.FloatTensor: [B, D] span representation with size D
        """
        B, L, D = token_embeddings.shape
        device = token_embeddings.device

        # Create token position grid: [1, L]
        positions = torch.arange(L, device=device).unsqueeze(0)

        # Build span mask: [B, L]
        span_mask = (positions >= left.unsqueeze(1)) & (positions <= right.unsqueeze(1))

        span_mask = span_mask.float()

        # Avoid division by zero (for case if noised span has 0 length)
        lengths = span_mask.sum(dim=1, keepdim=True).clamp(min=1.0)  # [B, 1]

        # Weighted sum → mean
        span_sum = torch.bmm(span_mask.unsqueeze(1), token_embeddings).squeeze(
            1
        )  # [B, D]

        span_mean = span_sum / lengths  # [B, D]

        span_mean = self.lin(span_mean)

        return span_mean


class TokenSelfAttention(nn.Module):
    """Multi-layer self-attention over tokens with pre-norm. Output [B, L, D]."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleDict({
                    "norm": nn.LayerNorm(embed_dim),
                    "self_attn": nn.MultiheadAttention(
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        dropout=dropout,
                        batch_first=True,
                    ),
                })
            )

    def forward(
        self,
        token_embeddings: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        # token_embeddings: [B, L, D]; key_padding_mask: [B, L], True = ignore
        x = token_embeddings
        for layer in self.layers:
            h = layer["norm"](x)
            x = x + layer["self_attn"](
                h, h, h,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )[0]
        return x


class SpanTokenCrossAttention(nn.Module):
    """Multi-layer shared cross-attention with pre-norm: query=span, kv=tokens."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleDict({
                    "norm": nn.LayerNorm(embed_dim),
                    "cross_attn": nn.MultiheadAttention(
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        dropout=dropout,
                        batch_first=False,
                    ),
                })
            )

    def forward(
        self,
        span: torch.Tensor,
        token_embeddings: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        # span: [B, D], token_embeddings: [B, L, D]; MultiheadAttention wants (L,B,D)
        kv = token_embeddings.transpose(0, 1)  # [L, B, D]
        x = span
        for layer in self.layers:
            q = layer["norm"](x).unsqueeze(0)  # [1, B, D]
            out, _ = layer["cross_attn"](
                q, kv, kv,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
            x = x + out.squeeze(0)
        return x


class DiffusionNERDenoiser(nn.Module, BaseDenoiser):
    def __init__(
        self,
        label_mapper: FloatIndexMapper,
        embedder_dim: int,
        num_steps: int,
        cross_attn_heads: int = 8,
        cross_attn_layers: int = 1,
        cross_attn_dropout: float = 0.1,
        self_attn_heads: int = 8,
        self_attn_layers: int = 1,
        self_attn_dropout: float = 0.1,
    ):
        super().__init__()
        self.label_mapper = label_mapper
        self.token_self_attn = TokenSelfAttention(
            embed_dim=embedder_dim,
            num_heads=self_attn_heads,
            num_layers=self_attn_layers,
            dropout=self_attn_dropout,
        )
        self.subject_span_block = SpanQueryBlock(embedder_dim, embedder_dim)
        self.object_span_block = SpanQueryBlock(embedder_dim, embedder_dim)
        self.predicate_span_block = SpanQueryBlock(embedder_dim, embedder_dim)
        self.span_token_cross_attn = SpanTokenCrossAttention(
            embed_dim=embedder_dim,
            num_heads=cross_attn_heads,
            num_layers=cross_attn_layers,
            dropout=cross_attn_dropout,
        )
        self.time_embedding = nn.Embedding(num_steps, embedder_dim)

        self.out = nn.Sequential(
            nn.Linear(4 * embedder_dim, 3 * embedder_dim),
            nn.ReLU(),
            nn.Linear(3 * embedder_dim, 6),
        )

    def forward(
        self,
        x_t: torch.FloatTensor,
        t: torch.LongTensor,
        condition: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.FloatTensor:
        """Denoising step

        Args:
            x_t (torch.FloatTensor): [B, 6]
            t (torch.LongTensor): [B]
            condition (tuple[torch.Tensor, torch.Tensor]): token embs and attn mask

        Returns:
            torch.FloatTensor: _description_
        """
        token_embeddings, attn_mask = condition
        # key_padding_mask: True = ignore (padding)
        key_padding_mask = (attn_mask == 0) if attn_mask is not None else None

        # Trainable encoding via self-attention over tokens
        token_encoding = self.token_self_attn(token_embeddings, key_padding_mask)

        sentence_len = attn_mask.sum(dim=1).long()  # [B]
        spans = self.label_mapper.reverse(x_t, sentence_len)  # [B, 6]
        (s_l, s_r, o_l, o_r, p_l, p_r) = spans.unbind(dim=1)
        s_span = self.subject_span_block(s_l, s_r, token_encoding)
        o_span = self.object_span_block(o_l, o_r, token_encoding)
        p_span = self.predicate_span_block(p_l, p_r, token_encoding)

        # Shared cross-attention: each span attends to token_encoding
        s_span = self.span_token_cross_attn(s_span, token_encoding, key_padding_mask)
        o_span = self.span_token_cross_attn(o_span, token_encoding, key_padding_mask)
        p_span = self.span_token_cross_attn(p_span, token_encoding, key_padding_mask)

        time_emb = self.time_embedding(t)
        x = torch.cat([s_span, o_span, p_span, time_emb], dim=1)
        return self.out(x)

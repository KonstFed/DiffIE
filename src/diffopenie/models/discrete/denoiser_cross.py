"""Cross-attention denoiser for discrete diffusion.

Architecture:
  - Input layer: tag-stream is primed per-position with the sentence-aligned
    slice of the projected BERT context (positional info comes from BERT, so
    no separate positional embedding is needed).
  - Per block: self-attention over the tag stream, then cross-attention to the
    prev-triplet slice of the context only (skipped when no prev triplets),
    then an FFN.

Shapes:
  - x_t:                    (B, L_s)              state ids
  - t:                      (B,)                  timestep
  - context:                (B, L_c, ctx_dim)     BERT embeddings of
                                                  sentence (+ prev-triplet text)
  - context_attention_mask: (B, L_c)              1 = real BERT token, 0 = pad
  - tag_attention_mask:     (B, L_s)              1 = valid tag, 0 = pad

Data invariant: the first L_s positions of `context` are the sentence tokens,
aligned 1:1 with the tag sequence. Any prev-triplet text occupies positions
[L_s, L_c). When L_c == L_s, there are no prev triplets and cross-attn is
skipped.
"""

from typing import Literal

import torch
from pydantic import BaseModel, ConfigDict, model_validator
from torch import nn

from diffopenie.models.discrete.denoiser import (
    LearnableTimeEmbedding,
    SinusoidalTimeEmbedding,
)


class IterativeBlock(nn.Module):
    """Pre-LN block: self-attn -> optional cross-attn -> FFN."""

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.ln_self = nn.LayerNorm(model_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ln_cross_q = nn.LayerNorm(model_dim)
        self.ln_cross_kv = nn.LayerNorm(model_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ln_ffn = nn.LayerNorm(model_dim)
        hidden = int(model_dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, model_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,                          # (B, L_s, D)
        prev_ctx: torch.Tensor | None,            # (B, L_p, D) or None
        tag_pad_mask: torch.Tensor,               # (B, L_s) True at PAD
        prev_pad_mask: torch.Tensor | None,       # (B, L_p) True at PAD, or None
    ) -> torch.Tensor:
        # 1. Self-attention over the tag stream.
        h = self.ln_self(x)
        attn_out = torch.zeros_like(h)
        valid_rows = ~tag_pad_mask.all(dim=1)
        if valid_rows.any():
            safe_out, _ = self.self_attn(
                h[valid_rows],
                h[valid_rows],
                h[valid_rows],
                key_padding_mask=tag_pad_mask[valid_rows],
                need_weights=False,
            )
            attn_out[valid_rows] = safe_out.to(attn_out.dtype)
        x = x + attn_out

        # 2. Cross-attention to prev triplets only — skipped when none.
        if prev_ctx is not None and prev_ctx.size(1) > 0:
            q = self.ln_cross_q(x)
            kv = self.ln_cross_kv(prev_ctx)
            attn_out = torch.zeros_like(q)
            # prev_pad_mask is guaranteed non-None when prev_ctx has positions
            valid_rows = ~prev_pad_mask.all(dim=1)
            if valid_rows.any():
                safe_out, _ = self.cross_attn(
                    q[valid_rows],
                    kv[valid_rows],
                    kv[valid_rows],
                    key_padding_mask=prev_pad_mask[valid_rows],
                    need_weights=False,
                )
                attn_out[valid_rows] = safe_out.to(attn_out.dtype)
            x = x + attn_out

        # 3. FFN.
        x = x + self.ffn(self.ln_ffn(x))
        return x


class CrossAttentionDenoiser(nn.Module):
    """Discrete diffusion denoiser with optional cross-attn to prev triplets.

    The tag stream is primed per-position with the sentence-aligned slice of
    the BERT context (so positional info comes from BERT directly). Each
    block self-attends over the tag stream and optionally cross-attends to
    the prev-triplet portion of the context.
    """

    def __init__(
        self,
        *,
        num_states: int,             # K
        model_dim: int,              # D
        ctx_dim: int,                # D_ctx
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.0,
        time_embed: str = "cosine",  # "cosine", "learnable", or "none"
        num_timesteps: int | None = None,  # required when time_embed=="learnable"
    ):
        super().__init__()
        if time_embed not in {"cosine", "learnable", "none"}:
            raise ValueError("time_embed must be 'cosine', 'learnable', or 'none'")
        if time_embed == "learnable" and (
            num_timesteps is None or num_timesteps <= 0
        ):
            raise ValueError(
                "num_timesteps must be set and > 0 when time_embed == 'learnable'"
            )

        self.num_states = num_states
        self.model_dim = model_dim
        self.ctx_dim = ctx_dim
        self.time_embed_type = time_embed

        self.state_embed = nn.Embedding(num_states, model_dim)

        if time_embed == "cosine":
            self.time_embed = SinusoidalTimeEmbedding(model_dim)
        elif time_embed == "learnable":
            self.time_embed = LearnableTimeEmbedding(num_timesteps, model_dim)
        else:
            self.time_embed = None

        self.ctx_proj = nn.Linear(ctx_dim, model_dim)

        self.input_ln = nn.LayerNorm(model_dim)
        self.input_dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [
                IterativeBlock(model_dim, num_heads, dropout=dropout)
                for _ in range(num_layers)
            ]
        )
        self.final_ln = nn.LayerNorm(model_dim)
        self.to_logits = nn.Linear(model_dim, num_states)

    def forward(
        self,
        x_t: torch.LongTensor,                  # (B, L_s)
        t: torch.LongTensor,                    # (B,)
        context: torch.Tensor,                  # (B, L_c, ctx_dim)
        context_attention_mask: torch.Tensor,   # (B, L_c) 1=keep, 0=pad
        tag_attention_mask: torch.Tensor,       # (B, L_s) 1=keep, 0=pad
    ) -> torch.Tensor:
        B, L_s = x_t.shape
        if context.shape[0] != B or context.shape[2] != self.ctx_dim:
            raise ValueError(
                f"context shape {tuple(context.shape)} doesn't match "
                f"(B={B}, L_c, ctx_dim={self.ctx_dim})"
            )
        L_c = context.shape[1]
        if L_s > L_c:
            raise ValueError(
                f"tag length {L_s} must be <= context length {L_c}; "
                "the first L_s context positions must be the sentence tokens."
            )
        if context_attention_mask.shape != (B, L_c):
            raise ValueError(
                f"context_attention_mask shape {tuple(context_attention_mask.shape)} != ({B}, {L_c})"
            )
        if tag_attention_mask.shape != (B, L_s):
            raise ValueError(
                f"tag_attention_mask shape {tuple(tag_attention_mask.shape)} != ({B}, {L_s})"
            )

        ctx = self.ctx_proj(context)  # (B, L_c, D)

        # Input layer: prime tag stream per-position with aligned sentence context.
        x = self.state_embed(x_t) + ctx[:, :L_s, :]  # (B, L_s, D)
        if self.time_embed is not None:
            x = x + self.time_embed(t - 1).unsqueeze(1)  # (B, 1, D)
        x = self.input_dropout(self.input_ln(x))

        # Prev-triplet slice for cross-attn.
        if L_c > L_s:
            prev_ctx = ctx[:, L_s:, :]                              # (B, L_p, D)
            prev_pad_mask = context_attention_mask[:, L_s:] == 0    # (B, L_p)
        else:
            prev_ctx = None
            prev_pad_mask = None

        tag_pad_mask = tag_attention_mask == 0  # (B, L_s) True at PAD

        for blk in self.blocks:
            x = blk(x, prev_ctx, tag_pad_mask, prev_pad_mask)

        x = self.final_ln(x)
        return self.to_logits(x)  # (B, L_s, K)


class CrossAttentionDenoiserConfig(BaseModel):
    """Configuration for CrossAttentionDenoiser."""

    model_config = ConfigDict(extra="forbid")
    type: Literal["cross_attn"] = "cross_attn"
    num_states: int
    model_dim: int
    ctx_dim: int
    num_layers: int = 4
    num_heads: int = 4
    dropout: float = 0.0
    time_embed: str = "cosine"
    num_timesteps: int | None = None

    @model_validator(mode="after")
    def _check_learnable_timesteps(self) -> "CrossAttentionDenoiserConfig":
        if self.time_embed == "learnable" and (
            self.num_timesteps is None or self.num_timesteps <= 0
        ):
            raise ValueError(
                "num_timesteps must be set and > 0 when time_embed == 'learnable'"
            )
        return self

    def create(self) -> CrossAttentionDenoiser:
        return CrossAttentionDenoiser(
            num_states=self.num_states,
            model_dim=self.model_dim,
            ctx_dim=self.ctx_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            dropout=self.dropout,
            time_embed=self.time_embed,
            num_timesteps=self.num_timesteps,
        )

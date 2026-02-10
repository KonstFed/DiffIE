# acutally IPED inspired. TODO: sort it out

import torch
import torch.nn as nn
from diffopenie.diffusion.scheduler import BaseDenoiser
from diffopenie.models.span.label_mapper import FloatIndexMapper


class SpanBlock(nn.Module):
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


class DiffusionNERDenoiser(nn.Module, BaseDenoiser):
    def __init__(self, label_mapper: FloatIndexMapper, embedder_dim: int, span_dim: int,):
        super().__init__()
        self.label_mapper = label_mapper
        self.subject_span_block = SpanBlock(embedder_dim, span_dim)
        self.object_span_block = SpanBlock(embedder_dim, span_dim)
        self.predicate_span_block = SpanBlock(embedder_dim, span_dim)

        self.out = nn.Sequential(
            nn.Linear(3 * span_dim, 3 * span_dim),
            nn.ReLU(),
            nn.Linear(3 * span_dim, 6),
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
        sentence_len = attn_mask.sum(dim=1).long()  # [B]
        spans = self.label_mapper.reverse(x_t, sentence_len)  # [B, 6]
        (s_l, s_r, o_l, o_r, p_l, p_r) = spans.unbind(dim=1)
        s_span = self.subject_span_block(s_l, s_r, token_embeddings)
        o_span = self.object_span_block(o_l, o_r, token_embeddings)
        p_span = self.predicate_span_block(p_l, p_r, token_embeddings)
        x = torch.cat([s_span, o_span, p_span], dim=1)
        return self.out(x)

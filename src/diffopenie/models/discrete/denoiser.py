import math
import torch
from torch import nn
from pydantic import BaseModel, ConfigDict


class SinusoidalTimeEmbedding(nn.Module):
    """Classic sinusoidal embedding for timesteps, then projected to model_dim."""
    def __init__(self, model_dim: int, max_period: int = 10_000):
        super().__init__()
        self.model_dim = model_dim
        self.max_period = max_period
        self.proj = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )

    def forward(self, t: torch.LongTensor) -> torch.Tensor:
        """
        Args:
            t: (B,) integer timesteps

        Returns:
            (B, model_dim) time embedding
        """
        half = self.model_dim // 2
        device = t.device
        # frequencies
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(0, half, device=device).float() / half
        )  # (half,)
        args = t.float()[:, None] * freqs[None, :]  # (B, half)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # (B, 2*half)
        if self.model_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros((t.size(0), 1), device=device)], dim=-1)
        return self.proj(emb)


class TransformerDenoiserBlock(nn.Module):
    """Pre-LN Transformer encoder block with attention mask support."""
    def __init__(self, model_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(model_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=model_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.ln2 = nn.LayerNorm(model_dim)
        hidden = int(model_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, model_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None) -> torch.Tensor:
        # key_padding_mask: (B, L) with True at PAD positions (PyTorch convention)
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x


class DiscreteDenoiser(nn.Module):
    """
    Baseline self-attention denoiser for discrete diffusion (D3PM-style).

    Inputs:
      - x_t: (B, L) integer noised states at time t
      - t:   (B,) timestep
      - token_embeddings: (B, L, D_ctx) external embeddings (e.g., from a pretrained encoder)
      - attention_mask: (B, L) with 1 for real tokens, 0 for padding

    Output:
      - logits over states: (B, L, K) = unnormalized p_theta(x0 | x_t, t)

    Fusion options:
      - fuse="sum":  project x_t embedding and token_embeddings into same dim and add
      - fuse="concat": concatenate then project
    """
    def __init__(
        self,
        *,
        num_states: int,          # K
        model_dim: int,           # D
        ctx_dim: int,             # D_ctx (dimension of token_embeddings)
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.0,
        fuse: str = "sum",        # "sum" or "concat"
    ):
        super().__init__()
        if fuse not in {"sum", "concat"}:
            raise ValueError("fuse must be 'sum' or 'concat'")

        self.num_states = num_states
        self.model_dim = model_dim
        self.fuse = fuse

        # Embed discrete states x_t (token ids in {0..K-1})
        self.state_embed = nn.Embedding(num_states, model_dim)

        # Time embedding (B, D) then broadcast to (B, L, D)
        self.time_embed = SinusoidalTimeEmbedding(model_dim)

        # Project external token embeddings into model_dim (always)
        self.ctx_proj = nn.Linear(ctx_dim, model_dim)

        # Fusion projection (only needed for concat)
        if fuse == "concat":
            self.fuse_proj = nn.Linear(2 * model_dim, model_dim)
        else:
            self.fuse_proj = None

        # Transformer encoder stack
        self.blocks = nn.ModuleList(
            [TransformerDenoiserBlock(model_dim, num_heads, dropout=dropout) for _ in range(num_layers)]
        )
        self.final_ln = nn.LayerNorm(model_dim)

        # Output head: logits over x0 states
        self.to_logits = nn.Linear(model_dim, num_states)

    def forward(
        self,
        x_t: torch.LongTensor,           # (B, L)
        t: torch.LongTensor,             # (B,)
        token_embeddings: torch.Tensor,  # (B, L, ctx_dim)
        attention_mask: torch.Tensor,    # (B, L) 1=keep, 0=pad
    ) -> torch.Tensor:
        B, L = x_t.shape
        if token_embeddings.shape[:2] != (B, L):
            raise ValueError("token_embeddings must have shape (B, L, ctx_dim)")
        if attention_mask.shape != (B, L):
            raise ValueError("attention_mask must have shape (B, L)")

        # PyTorch MHA uses key_padding_mask with True at pads
        key_padding_mask = (attention_mask == 0)  # (B, L) bool

        x_state = self.state_embed(x_t)               # (B, L, D)
        x_ctx = self.ctx_proj(token_embeddings)       # (B, L, D)
        t_emb = self.time_embed(t).unsqueeze(1)       # (B, 1, D)

        if self.fuse == "sum":
            h = x_state + x_ctx + t_emb               # broadcast over L
        else:
            h = torch.cat([x_state + t_emb, x_ctx], dim=-1)  # (B, L, 2D)
            h = self.fuse_proj(h)                     # (B, L, D)

        for blk in self.blocks:
            h = blk(h, key_padding_mask=key_padding_mask)

        h = self.final_ln(h)
        logits = self.to_logits(h)                    # (B, L, K)
        return logits


class DiscreteDenoiserConfig(BaseModel):
    """
    Configuration model for DiscreteDenoiser.
    Acts as a factory for creating DiscreteDenoiser instances.
    """

    model_config = ConfigDict(extra="forbid")
    num_states: int   # K
    model_dim: int    # D
    ctx_dim: int      # D_ctx (dimension of token_embeddings)
    num_layers: int = 4
    num_heads: int = 4
    dropout: float = 0.0
    fuse: str = "sum"  # "sum" or "concat"

    def create(self) -> DiscreteDenoiser:
        """
        Factory method to create a DiscreteDenoiser instance.

        Returns:
            Instance of DiscreteDenoiser

        Example:
            config = DiscreteDenoiserConfig(
                num_states=256, model_dim=256, ctx_dim=768, num_layers=4
            )
            denoiser = config.create()
        """
        return DiscreteDenoiser(
            num_states=self.num_states,
            model_dim=self.model_dim,
            ctx_dim=self.ctx_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            dropout=self.dropout,
            fuse=self.fuse,
        )

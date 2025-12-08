import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaLayerNorm(nn.Module):
    """
    Adaptive LayerNorm: LN(x) with gamma, beta predicted from condition c.
    x:  [B, L, d_model]
    c:  [B, L, d_cond] (usually d_cond == d_model)
    """
    def __init__(self, d_model: int, d_cond: int):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_cond, 2 * d_model),
            nn.SiLU(),
            nn.Linear(2 * d_model, 2 * d_model),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # cond -> gamma, beta
        h = self.mlp(cond)                        # [B, L, 2*d_model]
        gamma, beta = h.chunk(2, dim=-1)          # [B, L, d_model] each

        x_norm = self.ln(x)                       # [B, L, d_model]
        return gamma * x_norm + beta              # [B, L, d_model]


class DiffusionSLTransformerBlock(nn.Module):
    """
    One Transformer block with:
    - self-attention on x
    - AdaLayerNorm conditioned on c_t
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, d_cond: int):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True,
        )

        self.ada_ln1 = AdaLayerNorm(d_model, d_cond)
        self.ada_ln2 = AdaLayerNorm(d_model, d_cond)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

        # optional learnable scalars (often used in diffusion transformers)
        self.scale_attn = nn.Parameter(torch.zeros(1))
        self.scale_ff = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        x: torch.Tensor,       # [B, L, d_model]
        cond: torch.Tensor,    # [B, L, d_cond]
        key_padding_mask: torch.Tensor | None = None,  # [B, L] True = pad
    ) -> torch.Tensor:
        # Self-attention branch
        h = self.ada_ln1(x, cond)
        attn_out, _ = self.self_attn(
            h, h, h,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + self.scale_attn * attn_out

        # FF branch
        h = self.ada_ln2(x, cond)
        ff_out = self.ff(h)
        x = x + self.scale_ff * ff_out

        return x


class DiffusionSLDenoiser(nn.Module):
    """
    DiffusionSL-style denoiser.

    Inputs:
        x_t:             [B, L, x_dim]           noisy label/bit representation
        t:               [B]                    timesteps (0..num_steps-1)
        token_embeddings:[B, L, bert_dim]       BERT (or other encoder) outputs
        attn_mask:       [B, L] (True = keep, False = pad) or None

    Output:
        x0_pred:         [B, L, x_dim]          predicted clean labels
    """
    def __init__(
        self,
        x_dim: int,           # dimension of x_t / x0
        bert_dim: int,        # dimension of BERT token embeddings
        num_steps: int = 1000,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        d_ff: int = 1024,
    ):
        super().__init__()

        self.d_model = d_model

        # --- Condition Embedder: c_t = Linear(Concat(H, Embeds(t))) ---
        self.time_emb = nn.Embedding(num_steps, d_model)
        self.cond_linear = nn.Linear(bert_dim + d_model, d_model)  # c_t has dim d_model

        # --- Pre Layer: project x_t (bits) to model space ---
        self.pre = nn.Sequential(
            nn.Linear(x_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # --- Generator: N Transformer blocks with AdaLayerNorm conditioning ---
        self.blocks = nn.ModuleList([
            DiffusionSLTransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                d_cond=d_model,
            )
            for _ in range(n_layers)
        ])

        # --- Post Layer: AdaLN + MLP back to x_dim ---
        self.post_ada_ln = AdaLayerNorm(d_model, d_model)
        self.post_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, x_dim),
        )

    def build_condition(
        self,
        token_embeddings: torch.Tensor,  # [B, L, bert_dim]
        t: torch.Tensor,                 # [B]
    ) -> torch.Tensor:
        """
        c_t = Linear(Concat(H, Embeds(t)))
        """
        B, L, _ = token_embeddings.shape

        t_emb = self.time_emb(t)               # [B, d_model]
        t_emb = t_emb.unsqueeze(1).expand(B, L, -1)  # [B, L, d_model]

        cond_input = torch.cat([token_embeddings, t_emb], dim=-1)  # [B, L, bert_dim + d_model]
        c_t = self.cond_linear(cond_input)                         # [B, L, d_model]
        return c_t

    def forward(
        self,
        x_t: torch.Tensor,             # [B, L, x_dim]
        t: torch.Tensor,               # [B]
        token_embeddings: torch.Tensor,# [B, L, bert_dim]
        attn_mask: torch.Tensor | None = None,  # [B, L] True=keep, False=pad
    ) -> torch.Tensor:
        B, L, _ = x_t.shape

        # Build conditional signal c_t
        c_t = self.build_condition(token_embeddings, t)   # [B, L, d_model]

        # Pre layer: noisy bits -> model space
        h = self.pre(x_t)                                 # [B, L, d_model]

        # Attention mask: MultiheadAttention uses key_padding_mask with True=pad
        if attn_mask is not None:
            key_padding_mask = ~attn_mask.bool()          # [B, L], True=pad
        else:
            key_padding_mask = None

        # Generator: Transformer stack with AdaLayerNorm conditioning
        for blk in self.blocks:
            h = blk(h, c_t, key_padding_mask=key_padding_mask)

        # Post layer: AdaLN + MLP back to x_dim
        h = self.post_ada_ln(h, c_t)                      # [B, L, d_model]
        x0_pred = self.post_mlp(h)                        # [B, L, x_dim]

        return x0_pred

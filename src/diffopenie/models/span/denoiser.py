from torch import nn
import torch

NUM_SLOTS = 6  # sub_l, sub_r, rel_l, rel_r, obj_l, obj_r

class SlotDecoderBlock(nn.Module):
    """
    Decoder block: refines Q via pre-norm + residual
    Q = Q + SelfAttn(LN(Q))
    Q = Q + CrossAttn(LN(Q), memory)
    Q = Q + FFN(LN(Q))

    Inputs: Q [B, 6, d_model], memory [B, L, d_model]
    Output: refined Q [B, 6, d_model]
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.ln_self = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True,
            dropout=dropout,
        )
        self.ln_cross = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True,
            dropout=dropout,
        )
        self.ln_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        Q: torch.Tensor,  # [B, 6, d_model]
        memory: torch.Tensor,  # [B, L, d_model]
        key_padding_mask: torch.Tensor | None = None,  # [B, L] True=pad
    ) -> torch.Tensor:
        # Self-attention over slots (pre-norm + residual)
        q_norm = self.ln_self(Q)
        attn_out, _ = self.self_attn(
            q_norm, q_norm, q_norm, need_weights=False
        )
        Q = Q + attn_out

        # Cross-attention: slots (query) -> sentence (key/value)
        q_norm = self.ln_cross(Q)
        cross_out, _ = self.cross_attn(
            q_norm,
            memory,
            memory,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        Q = Q + cross_out

        # FFN (pre-norm + residual)
        Q = Q + self.ffn(self.ln_ffn(Q))

        return Q


class SpanDenoiser(nn.Module):
    """
    One-step denoiser for simplex diffusion over 6 slot PMFs π_t ∈ R^{6×L}.

    Pipeline:
    1) C_t = π_t H; Q_0 = query_init_mlp([C_t; slot_id; time_emb])
    2) Q = stack of SlotDecoderBlock(Q, memory)  (refine Q through N layers)
    3) Pointer logits ℓ_t = W_o(Q) @ Kp^T; mask PAD positions
    """

    def __init__(
        self,
        bert_dim: int,
        num_steps: int = 1000,
        d_model: int | None = None,
        n_heads: int = 8,
        n_layers: int = 1,
        d_ff: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.bert_dim = bert_dim
        self.num_steps = num_steps
        self.d_model = d_model if d_model is not None else bert_dim
        self.n_layers = n_layers
        self.d_ff = d_ff if d_ff is not None else 4 * self.d_model

        # Slot identity embedding: s_j ∈ R^d for each of 6 slots
        self.slot_emb = nn.Embedding(NUM_SLOTS, self.d_model)
        self.time_emb = nn.Embedding(num_steps, self.d_model)

        # C_t from (π_t H) is bert_dim; project then concat with s_j, g_t
        self.c_proj = nn.Linear(bert_dim, self.d_model)
        # Initial Q from [C_t; slot_id; time_emb] — done once, then refined by blocks
        self.query_init_mlp = nn.Sequential(
            nn.Linear(3 * self.d_model, 2 * self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * self.d_model, self.d_model),
        )

        self.blocks = nn.ModuleList(
            [
                SlotDecoderBlock(
                    d_model=self.d_model,
                    n_heads=n_heads,
                    d_ff=self.d_ff,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

        # Pointer: Kp = W_p(H) used as memory and pointer keys; Up = W_o(Q)
        self.W_p = nn.Linear(bert_dim, self.d_model)  # H -> memory / Kp
        self.W_o = nn.Linear(self.d_model, self.d_model)  # Q -> Up

    def forward(
        self,
        x_t: torch.Tensor,  # [B, 6, L] slot PMFs π_t
        t: torch.Tensor,  # [B]
        condition: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """
        One denoising step: predict pointer logits for the 6 slots.

        Returns:
            logits: [B, 6, L] — π_{t-1} = softmax_row(logits) or residual update.
            PAD positions are masked with large negative value so they get no mass.
        """
        token_embeddings, attn_mask = condition
        B, _, L = x_t.shape
        H = token_embeddings  # [B, L, bert_dim]

        # Zero H at PAD so C_t and pointer keys get no contribution from padding
        if attn_mask is not None:
            H = H * attn_mask.unsqueeze(-1).to(H.dtype)

        # Safe timestep for nn.Embedding
        t = t.long().clamp(0, self.num_steps - 1)

        # Normalize x_t to PMF over positions (masked: PAD gets no mass)
        if attn_mask is not None:
            pad_mask = ~attn_mask.bool()  # [B, L], True = pad
            x_t_norm = x_t.masked_fill(pad_mask.unsqueeze(1), -1e9)
        else:
            x_t_norm = x_t
        x_t_norm = torch.softmax(x_t_norm, dim=-1)

        # 2.1 Slot content from PMFs: C_t = π_t H -> [B, 6, bert_dim]
        C_t = torch.bmm(x_t_norm, H)
        C_t = self.c_proj(C_t)  # [B, 6, d_model]

        # 2.2 Initial Q once: [C_t; slot_id; time_emb] -> Q [B, 6, d_model]
        slot_ids = torch.arange(
            NUM_SLOTS, device=x_t.device, dtype=torch.long
        )
        s = self.slot_emb(slot_ids).unsqueeze(0).expand(B, -1, -1)
        g_t = self.time_emb(t).unsqueeze(1).expand(B, NUM_SLOTS, -1)
        query_in = torch.cat([C_t, s, g_t], dim=-1)  # [B, 6, 3*d_model]
        Q = self.query_init_mlp(query_in)  # [B, 6, d_model]

        # Memory and pointer keys: same projection H_d = W_p(H)
        memory = self.W_p(H)  # [B, L, d_model] for cross-attn
        Kp = memory  # [B, L, d_model] for pointer logits

        key_padding_mask = None
        if attn_mask is not None:
            key_padding_mask = ~attn_mask.bool()  # [B, L], True = pad

        # 2.3 Stack of decoder blocks: refine Q
        for block in self.blocks:
            Q = block(Q, memory, key_padding_mask)

        # 2.4 Pointer logits: Up = W_o(Q), ℓ_t = Up @ Kp^T; mask PAD so they get no mass
        Up = self.W_o(Q)  # [B, 6, d_model]
        logits = torch.bmm(Up, Kp.transpose(1, 2))  # [B, 6, L]
        if attn_mask is not None:
            pad_mask = ~attn_mask.bool()  # [B, L], True = pad
            logits = logits.masked_fill(pad_mask.unsqueeze(1), -1e9)
        return logits

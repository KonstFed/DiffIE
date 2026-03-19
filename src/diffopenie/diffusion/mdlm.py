"""Masked Diffusion Language Model (MDLM) schedule.

Reference: Sahoo et al., "Simple and Effective Masked Diffusion Language Models", 2024.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict, Field

from diffopenie.diffusion.discrete import sample_categorical, to_one_hot
from diffopenie.diffusion.schedules import (
    AlphaScheduleConfig,
    CosineAlphaSchedule,
    CosineAlphaScheduleConfig,
)


class MDLMSchedule:
    """
    MDLM schedule for SMALL discrete state spaces (mask-absorbing only).

    Forward: each token independently masked with prob 1−α_t.
    Reverse: concrete score parameterization (unmask via model predictions).

    Compatible with D3PMSchedule interface: exposes sample_t, sample_forward,
    sample_reverse, forward_distribution, betas, forward_product, forward_transition.
    """

    def __init__(
        self,
        num_states: int,
        num_steps: int,
        mask_state_id: int,
        alphas: torch.Tensor | None = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.num_states = num_states
        self.num_steps = num_steps
        self.kernel = "mask_absorbing"
        self.mask_state_id = mask_state_id
        self.device = device
        self.dtype = dtype

        if not (0 <= mask_state_id < num_states):
            raise ValueError("mask_state_id must be in [0, num_states)")

        if alphas is None:
            alphas = CosineAlphaSchedule(
                num_steps=num_steps, device=device, dtype=dtype
            ).get_alphas()
        self.alphas = alphas.to(device, dtype)
        assert self.alphas.shape == (num_steps + 1,), (
            f"alphas must have shape ({num_steps + 1},), got {self.alphas.shape}"
        )

        # Derived: betas from alphas for compatibility
        # β_t = 1 − α_t / α_{t-1}
        self.betas = (1.0 - self.alphas[1:] / self.alphas[:-1].clamp_min(1e-10)).clamp(
            1e-6, 0.999
        )

        # Build D3PM-compatible matrices (cheap for small K)
        self.forward_transition = self._build_forward_transition()
        self.forward_product = self._build_forward_product()

    # ----------------------------
    # Compatibility matrices (small K)
    # ----------------------------

    def _build_forward_transition(self) -> torch.Tensor:
        """Build (T, K, K) per-step transition matrices from betas."""
        K = self.num_states
        m = self.mask_state_id
        I = torch.eye(K, device=self.device, dtype=self.dtype)
        Qs = []
        for t in range(self.num_steps):
            beta = self.betas[t]
            Q_t = (1.0 - beta) * I
            Q_t[:, m] += beta
            Q_t[m, :] = 0.0
            Q_t[m, m] = 1.0
            Qs.append(Q_t)
        return torch.stack(Qs, dim=0)

    def _build_forward_product(self) -> torch.Tensor:
        """
        Build (T+1, K, K) cumulative forward products from alphas.
        forward_product[t, k, j] = P(x_t = j | x_0 = k).
        """
        K = self.num_states
        T = self.num_steps
        m = self.mask_state_id

        fp = torch.zeros(T + 1, K, K, device=self.device, dtype=self.dtype)
        for t in range(T + 1):
            a = self.alphas[t].item()
            for k in range(K):
                if k == m:
                    fp[t, k, k] = 1.0
                else:
                    fp[t, k, k] = a
                    fp[t, k, m] = 1.0 - a
        return fp

    def to(self, device: str | torch.device) -> MDLMSchedule:
        if isinstance(device, torch.device):
            device_str = device.type
        else:
            device_str = str(device)
        self.device = device_str
        self.alphas = self.alphas.to(device_str, dtype=self.dtype)
        self.betas = self.betas.to(device_str, dtype=self.dtype)
        self.forward_transition = self.forward_transition.to(
            device_str, dtype=self.dtype
        )
        self.forward_product = self.forward_product.to(device_str, dtype=self.dtype)
        return self

    # ----------------------------
    # Sampling: t ~ p(t)
    # ----------------------------

    def sample_t(self, B: int) -> torch.LongTensor:
        """Uniform timestep sampling in {1..T}."""
        return torch.randint(
            1,
            self.num_steps + 1,
            size=(B,),
            device=self.device,
            dtype=torch.long,
        )

    def weight(self, t: torch.LongTensor) -> torch.Tensor:
        """NELBO per-timestep weight: w(t) = (α_{t-1} − α_t) / (1 − α_t)."""
        alpha_t = self.alphas[t].float()
        alpha_tm1 = self.alphas[t - 1].float()
        return ((alpha_tm1 - alpha_t) / (1.0 - alpha_t).clamp_min(1e-10)).clamp_min(0.0)

    # ----------------------------
    # Forward: q(x_t | x_0)
    # ----------------------------

    @torch.no_grad()
    def forward_distribution(
        self,
        x0: torch.LongTensor,
        t: torch.LongTensor,
    ) -> torch.Tensor:
        """
        q(x_t | x_0) via independent masking.

        For each position where x_0 = k (k ≠ MASK):
            P(x_t = k) = α_t,  P(x_t = MASK) = 1 − α_t
        For x_0 = MASK:
            P(x_t = MASK) = 1

        Returns (B, L, K).
        """
        B, L = x0.shape
        K = self.num_states
        m = self.mask_state_id

        alpha_t = self.alphas[t].view(B, 1, 1)  # (B, 1, 1)
        x0_oh = to_one_hot(x0, K).to(self.device, self.dtype)  # (B, L, K)

        is_mask = (x0 == m).unsqueeze(-1)  # (B, L, 1)

        probs = torch.zeros(B, L, K, device=self.device, dtype=self.dtype)
        probs.scatter_(-1, x0.unsqueeze(-1), alpha_t.expand(B, L, 1))
        probs[..., m] += 1.0 - alpha_t.squeeze(-1)  # (B, L)

        # If x_0 is MASK, force P(MASK)=1
        mask_dist = torch.zeros(B, L, K, device=self.device, dtype=self.dtype)
        mask_dist[..., m] = 1.0
        probs = torch.where(is_mask, mask_dist, probs)
        return probs

    @torch.no_grad()
    def sample_forward(
        self,
        x0: torch.LongTensor,
        t: torch.LongTensor,
    ) -> torch.LongTensor:
        """
        Sample x_t ~ q(x_t | x_0) by independent token masking.

        Each non-MASK token is replaced with MASK with probability 1−α_t.
        """
        B, L = x0.shape
        m = self.mask_state_id
        alpha_t = self.alphas[t].view(B, 1)  # (B, 1)

        survive = torch.rand(B, L, device=self.device) < alpha_t
        is_already_mask = x0 == m
        survive = survive | is_already_mask  # MASK tokens always stay MASK
        x_t = torch.where(survive, x0, torch.full_like(x0, m))
        return x_t

    # ----------------------------
    # Reverse: p_θ(x_{t-1} | x_t)
    # ----------------------------

    @torch.no_grad()
    def _reverse_distribution(
        self,
        x_t: torch.LongTensor,
        t: torch.LongTensor,
        p_x0_given_xt: torch.Tensor,
    ) -> torch.Tensor:
        """
        MDLM concrete score reverse distribution.

        For masked positions:
            P(x_{t-1}=j) = (α_{t-1}−α_t)/(1−α_t) · p̃_θ(x_0=j)  for j≠MASK
            P(x_{t-1}=MASK) = (1−α_{t-1})/(1−α_t)
        where p̃_θ is p_θ with mask state zeroed out and renormalized.

        For unmasked positions: P(x_{t-1} = x_t) = 1.
        """
        B, L = x_t.shape
        K = self.num_states
        m = self.mask_state_id

        alpha_t = self.alphas[t].float()  # (B,)
        alpha_tm1 = self.alphas[t - 1].float()  # (B,)
        one_minus_at = (1.0 - alpha_t).clamp_min(1e-10)

        unmask_w = ((alpha_tm1 - alpha_t) / one_minus_at).clamp(0, 1).view(B, 1, 1)
        stay_mask_w = ((1.0 - alpha_tm1) / one_minus_at).clamp(0, 1).view(B, 1)

        # Zero out mask state in predictions, renormalize
        p_clean = p_x0_given_xt.clone()
        p_clean[..., m] = 0.0
        p_clean = p_clean / p_clean.sum(-1, keepdim=True).clamp_min(1e-10)

        # Distribution for masked positions
        dist_masked = unmask_w * p_clean  # (B, L, K); index m is 0 from p_clean
        dist_masked[..., m] = stay_mask_w.expand(B, L)

        # Distribution for unmasked positions: delta at x_t
        dist_unmasked = to_one_hot(x_t, K).to(p_clean.dtype)

        is_masked = (x_t == m).unsqueeze(-1)  # (B, L, 1)
        return torch.where(is_masked, dist_masked, dist_unmasked)

    @torch.no_grad()
    def sample_reverse(
        self,
        x_t: torch.LongTensor,
        t: torch.LongTensor,
        p_x0_given_xt: torch.Tensor,
        argmax: bool = False,
    ) -> torch.LongTensor:
        """
        Sample x_{t-1} ~ p_θ(x_{t-1} | x_t) using MDLM concrete score.
        """
        probs = self._reverse_distribution(x_t, t, p_x0_given_xt)
        if argmax:
            return probs.argmax(dim=-1)
        return sample_categorical(probs)


# ============================================================
# Config
# ============================================================


class MDLMScheduleConfig(BaseModel):
    """
    Configuration for MDLMSchedule.
    Uses alpha_schedule subconfig (cosine | linear | log_linear | mi).
    """

    model_config = ConfigDict(extra="forbid")
    type: Literal["mdlm"] = "mdlm"
    num_states: int = 5
    num_steps: int = 128
    mask_state_id: int = 4
    device: str = "cpu"
    dtype: Literal["float32", "float16", "bfloat16"] = "float32"
    alpha_schedule: AlphaScheduleConfig = Field(
        default_factory=CosineAlphaScheduleConfig,
        description="Subconfig: cosine | linear | log_linear | mi with type-specific params.",
    )

    def create(self) -> MDLMSchedule:
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        dt = dtype_map[self.dtype]
        alphas = self.alpha_schedule.get_alphas(
            num_steps=self.num_steps,
            device=self.device,
            dtype=dt,
        )
        return MDLMSchedule(
            num_states=self.num_states,
            num_steps=self.num_steps,
            mask_state_id=self.mask_state_id,
            alphas=alphas,
            device=self.device,
            dtype=dt,
        )

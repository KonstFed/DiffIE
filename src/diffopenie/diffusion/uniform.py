"""Uniform discrete diffusion schedule.

Forward kernel: Q_t = (1 - β_t) I + β_t U  where U_ij = 1/K.
Reverse: exact D3PM Eq.(4) enumeration — works for any transition kernel.
No absorbing/mask state; all K label states are treated symmetrically.
"""

from __future__ import annotations

from typing import Literal

import torch
from pydantic import BaseModel, ConfigDict, Field

from diffopenie.diffusion.discrete import sample_categorical, to_one_hot
from diffopenie.diffusion.schedules import (
    BetaScheduleConfig,
    CosineBetaSchedule,
    CosineBetaScheduleConfig,
)


class UniformSchedule:
    """
    Uniform discrete diffusion schedule for small state spaces.

    Forward: Q_t = (1 - β_t) I + β_t U   (U_ij = 1/K, uniform corruption)
    Reverse: D3PM Eq.(4) exact enumeration over x̂_0

    Compatible with D3PMSchedule / MDLMSchedule interface: exposes
    sample_t, sample_forward, sample_reverse, forward_distribution,
    betas, forward_transition, forward_product, kernel, mask_state_id.
    """

    def __init__(
        self,
        num_states: int,
        num_steps: int,
        pad_state_id: int | None = None,
        betas: torch.Tensor | None = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.num_states = num_states
        self.num_steps = num_steps
        self.kernel = "uniform"
        self.mask_state_id = None
        self.pad_state_id = pad_state_id  # positions with this label are never corrupted
        self.device = device
        self.dtype = dtype

        if betas is None:
            self.betas = CosineBetaSchedule(
                num_steps=num_steps, device=device, dtype=dtype
            ).get_betas()
        else:
            self.betas = betas.to(device, dtype)
            assert self.betas.shape == (num_steps,)
        self.betas = self.betas.clamp(1e-6, 0.999)

        self.forward_transition = self._build_forward_transition()  # (T, K, K)
        self.forward_product = self._build_cumulative_products()    # (T+1, K, K)

    def _build_forward_transition(self) -> torch.Tensor:
        K = self.num_states
        I = torch.eye(K, device=self.device, dtype=self.dtype)
        U = torch.full((K, K), 1.0 / K, device=self.device, dtype=self.dtype)
        Qs = []
        for t in range(self.num_steps):
            beta = self.betas[t]
            Qs.append((1.0 - beta) * I + beta * U)
        return torch.stack(Qs, dim=0)

    def _build_cumulative_products(self) -> torch.Tensor:
        K = self.num_states
        bar = [torch.eye(K, device=self.device, dtype=self.dtype)]
        cur = bar[0]
        for t in range(self.num_steps):
            cur = cur @ self.forward_transition[t]
            bar.append(cur)
        return torch.stack(bar, dim=0)

    def to(self, device: str | torch.device) -> UniformSchedule:
        device_str = device.type if isinstance(device, torch.device) else str(device)
        self.device = device_str
        self.betas = self.betas.to(device_str, dtype=self.dtype)
        self.forward_transition = self.forward_transition.to(device_str, dtype=self.dtype)
        self.forward_product = self.forward_product.to(device_str, dtype=self.dtype)
        return self

    # ----------------------------
    # Timestep sampling
    # ----------------------------

    def sample_t(self, B: int) -> torch.LongTensor:
        return torch.randint(
            1, self.num_steps + 1, size=(B,), device=self.device, dtype=torch.long
        )

    # ----------------------------
    # Forward: q(x_t | x_0)
    # ----------------------------

    @torch.no_grad()
    def forward_distribution(
        self, x0: torch.LongTensor, t: torch.LongTensor
    ) -> torch.Tensor:
        """
        q(x_t | x_0) = Cat( x_0 barQ_t )

        Args:
            x0: (B, L)
            t:  (B,) in {1..T}

        Returns:
            (B, L, K)
        """
        x0_oh = to_one_hot(x0, self.num_states).to(self.device, dtype=torch.float32)
        barQ_t = self.forward_product[t].float()
        # element-wise: avoids cuBLAS (which misbehaves under AMP for this GPU)
        # result[b,l,j] = sum_k x0_oh[b,l,k] * barQ_t[b,k,j]
        return (x0_oh.unsqueeze(-1) * barQ_t.unsqueeze(1)).sum(dim=2)

    @torch.no_grad()
    def sample_forward(
        self, x0: torch.LongTensor, t: torch.LongTensor
    ) -> torch.LongTensor:
        x_t = sample_categorical(self.forward_distribution(x0, t))
        if self.pad_state_id is not None:
            x_t = torch.where(x0 == self.pad_state_id, x0, x_t)
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
        Exact D3PM Eq.(4):
            p_θ(x_{t-1}|x_t) = Σ_{x̂0} q(x_{t-1}|x_t, x̂0) p_θ(x̂0|x_t)

        where q(x_{t-1}|x_t, x̂0) ∝ [x_t Q_t^T] ⊙ [x̂0 barQ_{t-1}], normalized per x̂0.
        """
        B, L = x_t.shape
        K = self.num_states

        x_t_oh = to_one_hot(x_t, K).to(self.device, dtype=torch.float32)
        p_x0_given_xt = p_x0_given_xt.to(self.device, dtype=torch.float32)

        Q_t = self.forward_transition[t - 1].float()    # (B, K, K)
        barQ_tm1 = self.forward_product[t - 1].float()  # (B, K, K)

        # a[b,l,j] = sum_k x_t_oh[b,l,k] * Q_t[b,j,k]  (Q_t transposed)
        a = (x_t_oh.unsqueeze(-1) * Q_t.transpose(-1, -2).unsqueeze(1)).sum(dim=2)

        out = torch.zeros(B, L, K, device=self.device, dtype=torch.float32)
        for k in range(K):
            # b_k[b,l,j] = barQ_tm1[b,k,j]  (one-hot at k picks the k-th row)
            b_k = barQ_tm1[:, k, :].unsqueeze(1).expand(B, L, K)
            qk_unnorm = a * b_k
            qk = qk_unnorm / qk_unnorm.sum(-1, keepdim=True).clamp_min(1e-12)
            out += p_x0_given_xt[..., k].unsqueeze(-1) * qk

        return out / out.sum(-1, keepdim=True).clamp_min(1e-12)

    @torch.no_grad()
    def sample_reverse(
        self,
        x_t: torch.LongTensor,
        t: torch.LongTensor,
        p_x0_given_xt: torch.Tensor,
        argmax: bool = False,
    ) -> torch.LongTensor:
        probs = self._reverse_distribution(x_t, t, p_x0_given_xt)
        if argmax:
            return probs.argmax(dim=-1)
        return sample_categorical(probs)


class UniformScheduleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["uniform"] = "uniform"
    num_states: int = 4
    num_steps: int = 64
    pad_state_id: int | None = 4  # label value that is never corrupted (sentence padding)
    device: str = "cpu"
    dtype: Literal["float32", "float16", "bfloat16"] = "float32"
    beta_schedule: BetaScheduleConfig = Field(
        default_factory=CosineBetaScheduleConfig,
        description="Subconfig: cosine | linear | log_linear | mi with type-specific params.",
    )

    def create(self) -> UniformSchedule:
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        dt = dtype_map[self.dtype]
        betas = self.beta_schedule.get_betas(
            num_steps=self.num_steps,
            device=self.device,
            dtype=dt,
        )
        return UniformSchedule(
            num_states=self.num_states,
            num_steps=self.num_steps,
            pad_state_id=self.pad_state_id,
            betas=betas,
            device=self.device,
            dtype=dt,
        )

"""Noise schedules for discrete diffusion models.

Beta schedules (D3PM): produce per-step betas β_1..β_T.
Alpha schedules (MDLM): produce survival probabilities α_0..α_T where α_0=1.
"""

import math
from typing import Annotated, Literal, Union

import torch
from pydantic import BaseModel, ConfigDict, Field


# ============================================================
# Beta schedules — produce (T,) tensor of per-step betas
# ============================================================


class CosineBetaSchedule:
    """
    Cosine noise schedule (Nichol & Dhariwal style).

    alpha_bar(t) = cos^2((t + s) / (1 + s) * pi/2),
    then beta_t = 1 - alpha_bar_t / alpha_bar_{t-1}.
    """

    def __init__(
        self,
        num_steps: int,
        s: float = 0.008,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.num_steps = num_steps
        self.s = s
        self.device = device
        self.dtype = dtype

    def get_betas(self) -> torch.Tensor:
        steps = torch.arange(
            1, self.num_steps + 1, device=self.device, dtype=self.dtype
        )
        t = steps / self.num_steps
        alpha_bar = torch.cos((t + self.s) / (1.0 + self.s) * (math.pi / 2.0)) ** 2
        alpha_bar_prev = torch.cat(
            [torch.tensor([1.0], device=self.device, dtype=self.dtype), alpha_bar[:-1]]
        )
        alpha = alpha_bar / alpha_bar_prev
        betas = (1.0 - alpha).clamp(1e-6, 0.999)
        return betas


class LinearBetaSchedule:
    """Linear interpolation from beta_start to beta_end."""

    def __init__(
        self,
        num_steps: int,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.num_steps = num_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = device
        self.dtype = dtype

    def get_betas(self) -> torch.Tensor:
        return torch.linspace(
            self.beta_start,
            self.beta_end,
            self.num_steps,
            device=self.device,
            dtype=self.dtype,
        )


class LogLinearBetaSchedule:
    """Log-linear: log(beta_t) is linear in t."""

    def __init__(
        self,
        num_steps: int,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.num_steps = num_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = device
        self.dtype = dtype

    def get_betas(self) -> torch.Tensor:
        if self.num_steps == 1:
            return torch.tensor([self.beta_start], device=self.device, dtype=self.dtype)
        log_start = math.log(max(self.beta_start, 1e-10))
        log_end = math.log(max(self.beta_end, 1e-10))
        u = torch.linspace(
            0.0, 1.0, self.num_steps, device=self.device, dtype=self.dtype
        )
        log_betas = (1.0 - u) * log_start + u * log_end
        return torch.exp(log_betas).clamp(1e-6, 0.999)


def mi_betas_absorbing(
    num_steps: int,
    s_T: float = 0.1,
    mode: str = "linear",
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    beta_min: float = 1e-8,
    beta_max: float = 0.2,
) -> torch.Tensor:
    """
    Mutual-information schedule for mask-absorbing diffusion.

    Controls survival probability s_t from 1 → s_T, then converts:
        beta_t = 1 - s_t / s_{t-1}
    """
    T = num_steps
    s_T = float(s_T)
    assert 0.0 < s_T < 1.0

    if mode == "linear":
        s = torch.linspace(1.0, s_T, steps=T + 1, device=device, dtype=dtype)
    elif mode == "cosine":
        u = torch.linspace(0.0, 1.0, steps=T + 1, device=device, dtype=dtype)
        s = s_T + (1.0 - s_T) * torch.cos(u * torch.pi / 2.0) ** 2
    else:
        raise ValueError("mode must be 'linear' or 'cosine'")

    betas = 1.0 - (s[1:] / s[:-1]).clamp_min(1e-30)
    betas = betas.clamp(beta_min, beta_max)
    return betas


class MIBetaSchedule:
    """Mutual-information schedule for mask-absorbing D3PM."""

    def __init__(
        self,
        num_steps: int,
        s_T: float = 0.1,
        mode: str = "linear",
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        beta_min: float = 1e-8,
        beta_max: float = 0.2,
    ):
        self.num_steps = num_steps
        self.s_T = s_T
        self.mode = mode
        self.device = device
        self.dtype = dtype
        self.beta_min = beta_min
        self.beta_max = beta_max

    def get_betas(self) -> torch.Tensor:
        return mi_betas_absorbing(
            num_steps=self.num_steps,
            s_T=self.s_T,
            mode=self.mode,
            device=self.device,
            dtype=self.dtype,
            beta_min=self.beta_min,
            beta_max=self.beta_max,
        )


# ------------------------------------------------------------
# Beta schedule Pydantic configs
# ------------------------------------------------------------


class CosineBetaScheduleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["cosine"] = "cosine"
    s: float = 0.008

    def get_betas(
        self, num_steps: int, device: str, dtype: torch.dtype
    ) -> torch.Tensor:
        return CosineBetaSchedule(
            num_steps=num_steps, s=self.s, device=device, dtype=dtype
        ).get_betas()


class LinearBetaScheduleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["linear"] = "linear"
    beta_start: float = 0.0001
    beta_end: float = 0.02

    def get_betas(
        self, num_steps: int, device: str, dtype: torch.dtype
    ) -> torch.Tensor:
        return LinearBetaSchedule(
            num_steps=num_steps,
            beta_start=self.beta_start,
            beta_end=self.beta_end,
            device=device,
            dtype=dtype,
        ).get_betas()


class LogLinearBetaScheduleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["log_linear"] = "log_linear"
    beta_start: float = 0.0001
    beta_end: float = 0.02

    def get_betas(
        self, num_steps: int, device: str, dtype: torch.dtype
    ) -> torch.Tensor:
        return LogLinearBetaSchedule(
            num_steps=num_steps,
            beta_start=self.beta_start,
            beta_end=self.beta_end,
            device=device,
            dtype=dtype,
        ).get_betas()


class MIBetaScheduleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["mi"] = "mi"
    s_T: float = 0.1
    mode: Literal["linear", "cosine"] = "linear"
    beta_min: float = 1e-8
    beta_max: float = 0.2

    def get_betas(
        self, num_steps: int, device: str, dtype: torch.dtype
    ) -> torch.Tensor:
        return mi_betas_absorbing(
            num_steps=num_steps,
            s_T=self.s_T,
            mode=self.mode,
            device=device,
            dtype=dtype,
            beta_min=self.beta_min,
            beta_max=self.beta_max,
        )


BetaScheduleConfig = Annotated[
    Union[
        CosineBetaScheduleConfig,
        LinearBetaScheduleConfig,
        LogLinearBetaScheduleConfig,
        MIBetaScheduleConfig,
    ],
    Field(discriminator="type"),
]


# ============================================================
# Alpha schedules — produce (T+1,) tensor of survival probs
#   α_0 = 1 (clean), α_T ≈ 0 (fully masked)
# ============================================================


class CosineAlphaSchedule:
    """
    Cosine alpha schedule: α(t) = cos²((t/T + s)/(1+s) · π/2), normalized so α(0)=1.
    """

    def __init__(
        self,
        num_steps: int,
        s: float = 0.008,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.num_steps = num_steps
        self.s = s
        self.device = device
        self.dtype = dtype

    def get_alphas(self) -> torch.Tensor:
        t = torch.arange(0, self.num_steps + 1, device=self.device, dtype=self.dtype)
        t = t / self.num_steps
        alphas = torch.cos((t + self.s) / (1.0 + self.s) * (math.pi / 2.0)) ** 2
        alphas = alphas / alphas[0]
        return alphas.clamp_min(1e-10)


class LinearAlphaSchedule:
    """Linear alpha: α(t) = 1 − (1−eps)·t/T."""

    def __init__(
        self,
        num_steps: int,
        eps: float = 1e-4,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.num_steps = num_steps
        self.eps = eps
        self.device = device
        self.dtype = dtype

    def get_alphas(self) -> torch.Tensor:
        t = torch.arange(0, self.num_steps + 1, device=self.device, dtype=self.dtype)
        t = t / self.num_steps
        return (1.0 - (1.0 - self.eps) * t).clamp_min(1e-10)


class LogLinearAlphaSchedule:
    """
    Log-linear alpha: α(t) = eps^(t/T).
    Standard MDLM schedule where log(α(t)) is linear in t.
    """

    def __init__(
        self,
        num_steps: int,
        eps: float = 1e-4,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.num_steps = num_steps
        self.eps = eps
        self.device = device
        self.dtype = dtype

    def get_alphas(self) -> torch.Tensor:
        t = torch.linspace(
            0.0, 1.0, self.num_steps + 1, device=self.device, dtype=self.dtype
        )
        log_alpha = math.log(max(self.eps, 1e-10)) * t
        return torch.exp(log_alpha).clamp_min(1e-10)


class MIAlphaSchedule:
    """
    Mutual-information controlled alpha schedule for absorbing diffusion.
    α(t) goes from 1 to s_T via linear or cosine interpolation.
    """

    def __init__(
        self,
        num_steps: int,
        s_T: float = 0.1,
        mode: str = "linear",
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.num_steps = num_steps
        self.s_T = float(s_T)
        self.mode = mode
        self.device = device
        self.dtype = dtype

    def get_alphas(self) -> torch.Tensor:
        T = self.num_steps
        if self.mode == "linear":
            return torch.linspace(
                1.0, self.s_T, steps=T + 1, device=self.device, dtype=self.dtype
            )
        elif self.mode == "cosine":
            u = torch.linspace(
                0.0, 1.0, steps=T + 1, device=self.device, dtype=self.dtype
            )
            return (
                self.s_T + (1.0 - self.s_T) * torch.cos(u * torch.pi / 2.0) ** 2
            ).clamp_min(1e-10)
        else:
            raise ValueError("mode must be 'linear' or 'cosine'")


# ------------------------------------------------------------
# Alpha schedule Pydantic configs
# ------------------------------------------------------------


class CosineAlphaScheduleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["cosine"] = "cosine"
    s: float = 0.008

    def get_alphas(
        self, num_steps: int, device: str, dtype: torch.dtype
    ) -> torch.Tensor:
        return CosineAlphaSchedule(
            num_steps=num_steps, s=self.s, device=device, dtype=dtype
        ).get_alphas()


class LinearAlphaScheduleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["linear"] = "linear"
    eps: float = 1e-4

    def get_alphas(
        self, num_steps: int, device: str, dtype: torch.dtype
    ) -> torch.Tensor:
        return LinearAlphaSchedule(
            num_steps=num_steps, eps=self.eps, device=device, dtype=dtype
        ).get_alphas()


class LogLinearAlphaScheduleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["log_linear"] = "log_linear"
    eps: float = 1e-4

    def get_alphas(
        self, num_steps: int, device: str, dtype: torch.dtype
    ) -> torch.Tensor:
        return LogLinearAlphaSchedule(
            num_steps=num_steps, eps=self.eps, device=device, dtype=dtype
        ).get_alphas()


class MIAlphaScheduleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["mi"] = "mi"
    s_T: float = 0.1
    mode: Literal["linear", "cosine"] = "linear"

    def get_alphas(
        self, num_steps: int, device: str, dtype: torch.dtype
    ) -> torch.Tensor:
        return MIAlphaSchedule(
            num_steps=num_steps,
            s_T=self.s_T,
            mode=self.mode,
            device=device,
            dtype=dtype,
        ).get_alphas()


AlphaScheduleConfig = Annotated[
    Union[
        CosineAlphaScheduleConfig,
        LinearAlphaScheduleConfig,
        LogLinearAlphaScheduleConfig,
        MIAlphaScheduleConfig,
    ],
    Field(discriminator="type"),
]

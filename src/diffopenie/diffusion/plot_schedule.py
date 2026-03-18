"""Plot survival / absorption curves for a mask-absorbing scheduler (D3PM or MDLM)."""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from diffopenie.diffusion.discrete import D3PMSchedule
from diffopenie.diffusion.mdlm import MDLMSchedule


def plot_survival_prob(scheduler: D3PMSchedule | MDLMSchedule, save_path: str | None = None):
    """
    Plot cumulative survival probability for each non-mask state in a
    mask_absorbing scheduler: P(x_t = x_0 | x_0 = k) for t = 0..T.
    """
    if scheduler.kernel != "mask_absorbing":
        raise ValueError("plot_survival_prob only supports kernel='mask_absorbing'")

    K = scheduler.num_states
    T = scheduler.num_steps
    m = scheduler.mask_state_id
    barQ = scheduler.forward_product  # (T+1, K, K)
    betas = scheduler.betas  # (T,)

    ts = list(range(T + 1))

    # Cumulative survival: ∏_{s=1..t} (1 - β_s), with t=0 -> 1
    cum_survival = [1.0] + torch.cumprod(1.0 - betas, dim=0).cpu().tolist()

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Left: per-state survival (diagonal of barQ) + cumulative ∏(1-β_t)
    ax = axes[0]
    for k in range(K):
        if k == m:
            continue
        surv = [barQ[t, k, k].item() for t in ts]
        ax.plot(ts, surv, label=f"state {k}")
    ax.plot(ts, 1 - torch.tensor(cum_survival), "k--", label=r"$1 - \prod_{s=1}^t (1-\beta_s)$", linewidth=2)
    ax.set_xlabel("timestep t")
    ax.set_ylabel("P(x_t = x_0 | x_0 = k)")
    ax.set_title("Survival probability (stay in original state)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: probability of being masked
    ax = axes[1]
    for k in range(K):
        if k == m:
            continue
        mask_prob = [barQ[t, k, m].item() for t in ts]
        ax.plot(ts, mask_prob, label=f"state {k}")
    ax.set_xlabel("timestep t")
    ax.set_ylabel("P(x_t = MASK | x_0 = k)")
    ax.set_title("Mask absorption probability")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    # For mask absorbing, P(x_t = MASK | x_0 = k) is same for all non-mask k
    mask_prob = torch.tensor([barQ[t, 0, m].item() for t in ts])
    mask_prob = F.softmax(torch.sqrt(mask_prob), dim=0)
    # mask_prob = mask_prob / mask_prob.sum()
    cdf = torch.cumsum(mask_prob, dim=0).cpu().numpy()
    ax.bar(ts, mask_prob.cpu().numpy(), label="softmax(prob)", alpha=0.7)
    # ax.plot(ts, cdf, "o-", color="C1", label="CDF", linewidth=2, markersize=4)
    ax.set_xlabel("timestep t")
    ax.set_ylabel("probability")
    ax.set_title("Softmax over t of P(x_t = MASK | x_0) and CDF")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")
    plt.show()


if __name__ == "__main__":
    from pydantic import BaseModel, ConfigDict, TypeAdapter

    from diffopenie.models.discrete.discrete_model import SchedulerConfig
    from diffopenie.utils import load_config

    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/training_mdlm.yaml"

    class _Cfg(BaseModel):
        model_config = ConfigDict(extra="allow")
        model: dict

    raw = load_config(_Cfg, config_path)
    adapter = TypeAdapter(SchedulerConfig)
    sched_cfg = adapter.validate_python(raw.model["scheduler"])
    scheduler = sched_cfg.create()
    sched_type = sched_cfg.type
    if hasattr(sched_cfg, "beta_schedule"):
        print(f"{sched_type} / beta_schedule: {sched_cfg.beta_schedule.type}")
    elif hasattr(sched_cfg, "alpha_schedule"):
        print(f"{sched_type} / alpha_schedule: {sched_cfg.alpha_schedule.type}")

    print(f"kernel={scheduler.kernel}  K={scheduler.num_states}  T={scheduler.num_steps}")
    print(f"betas: min={scheduler.betas.min():.6f}  max={scheduler.betas.max():.6f}")
    plot_survival_prob(scheduler, save_path="survival_prob.png")

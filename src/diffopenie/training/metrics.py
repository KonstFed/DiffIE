"""Torchmetrics-based metrics for triplet extraction training."""

from dataclasses import dataclass

import torch
from torchmetrics import Metric

CLASS_NAMES = ("bg", "subj", "rel", "obj")


@dataclass
class EpochResult:
    """Aggregated results from one training epoch."""

    loss: float
    direct_metrics: "MetricsResult"
    per_timestep_loss: torch.Tensor  # [T]


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f = 2 * p * r / max(p + r, 1e-8)
    return p, r, f


@dataclass(frozen=True)
class MetricsResult:
    """Token-overlap P/R/F1 for triplet extraction."""

    precision: float
    recall: float
    f1: float
    class_precision: tuple[float, ...]  # (bg, subj, rel, obj)
    class_recall: tuple[float, ...]
    class_f1: tuple[float, ...]

    def to_dict(self, prefix: str = "") -> dict[str, float]:
        d = {
            f"{prefix}precision": self.precision,
            f"{prefix}recall": self.recall,
            f"{prefix}f1": self.f1,
        }
        for i, name in enumerate(CLASS_NAMES):
            d[f"{prefix}precision_{name}"] = self.class_precision[i]
            d[f"{prefix}recall_{name}"] = self.class_recall[i]
            d[f"{prefix}f1_{name}"] = self.class_f1[i]
        return d


class TripletMetrics(Metric):
    """CaRB-style token-overlap P/R/F1.

    Per-class for B(0), S(1), R(2), O(3).
    Overall P/R/F1 micro-averaged over S, R, O only.
    """

    full_state_update = False

    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.num_classes = num_classes
        self.add_state(
            "tp", default=torch.zeros(num_classes, dtype=torch.long), dist_reduce_fx="sum"
        )
        self.add_state(
            "fp", default=torch.zeros(num_classes, dtype=torch.long), dist_reduce_fx="sum"
        )
        self.add_state(
            "fn", default=torch.zeros(num_classes, dtype=torch.long), dist_reduce_fx="sum"
        )

    def update(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ):
        preds, target = preds.flatten(), target.flatten()
        if mask is not None:
            valid = mask.flatten().bool()
            preds, target = preds[valid], target[valid]
        for c in range(self.num_classes):
            pc, tc = preds == c, target == c
            self.tp[c] += (pc & tc).sum()
            self.fp[c] += (pc & ~tc).sum()
            self.fn[c] += (~pc & tc).sum()

    def compute(self) -> MetricsResult:
        per_class = [
            _prf(self.tp[c].item(), self.fp[c].item(), self.fn[c].item())
            for c in range(self.num_classes)
        ]
        per_p, per_r, per_f = zip(*per_class)
        p, r, f = _prf(
            self.tp[1:].sum().item(),
            self.fp[1:].sum().item(),
            self.fn[1:].sum().item(),
        )
        return MetricsResult(
            precision=p, recall=r, f1=f,
            class_precision=per_p, class_recall=per_r, class_f1=per_f,
        )


class PerTimestepLoss(Metric):
    """Tracks average loss per diffusion timestep t."""

    full_state_update = False

    def __init__(self, num_steps: int):
        super().__init__()
        self.num_steps = num_steps
        self.add_state(
            "loss_sum", default=torch.zeros(num_steps), dist_reduce_fx="sum"
        )
        self.add_state(
            "count", default=torch.zeros(num_steps, dtype=torch.long), dist_reduce_fx="sum"
        )

    def update(self, per_sample_loss: torch.Tensor, timesteps: torch.Tensor):
        """per_sample_loss: [B], timesteps: [B] (1-indexed)."""
        t_idx = (timesteps - 1).long()
        self.loss_sum.scatter_add_(0, t_idx, per_sample_loss.detach().float())
        self.count.scatter_add_(0, t_idx, torch.ones_like(t_idx))

    def compute(self) -> torch.Tensor:
        """Returns [num_steps] average loss per timestep."""
        result = torch.zeros_like(self.loss_sum)
        valid = self.count > 0
        result[valid] = self.loss_sum[valid] / self.count[valid].float()
        return result

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
    t_sampled_counts: torch.Tensor | None = None  # [T], samples per t in this epoch


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f = 2 * p * r / max(p + r, 1e-8)
    return p, r, f


@dataclass
class MetricsResult:
    """Token-overlap P/R/F1 for triplet extraction."""

    precision: float
    recall: float
    f1: float
    class_precision: tuple[float, ...]  # (bg, subj, rel, obj)
    class_recall: tuple[float, ...]
    class_f1: tuple[float, ...]
    ratio_masked: float | None = None  # fraction of valid tokens still in mask state

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
        if self.ratio_masked is not None:
            d[f"{prefix}ratio_masked"] = self.ratio_masked
        return d


class TripletMetrics(Metric):
    """CaRB-style token-overlap P/R/F1.

    Per-class for B(0), S(1), R(2), O(3).
    Overall P/R/F1 micro-averaged over S, R, O only.
    If mask_state_id is set, also computes ratio of valid tokens still in mask state.
    """

    full_state_update = False

    def __init__(self, num_classes: int = 4, mask_state_id: int | None = None):
        super().__init__()
        self.num_classes = num_classes
        self.mask_state_id = mask_state_id
        self.add_state(
            "tp", default=torch.zeros(num_classes, dtype=torch.long), dist_reduce_fx="sum"
        )
        self.add_state(
            "fp", default=torch.zeros(num_classes, dtype=torch.long), dist_reduce_fx="sum"
        )
        self.add_state(
            "fn", default=torch.zeros(num_classes, dtype=torch.long), dist_reduce_fx="sum"
        )
        if mask_state_id is not None:
            self.add_state(
                "masked_count", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum"
            )
            self.add_state(
                "valid_count", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum"
            )

    def update(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ):
        preds_flat = preds.flatten()
        target_flat = target.flatten()
        valid = (
            mask.flatten().bool()
            if mask is not None
            else torch.ones_like(preds_flat, dtype=torch.bool, device=preds_flat.device)
        )
        if self.mask_state_id is not None:
            self.masked_count += ((preds_flat == self.mask_state_id) & valid).sum()
            self.valid_count += valid.sum()
        preds = preds_flat[valid]
        target = target_flat[valid]
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
        ratio_masked = None
        if self.mask_state_id is not None and hasattr(self, "valid_count"):
            v = self.valid_count.item()
            ratio_masked = self.masked_count.item() / max(v, 1)
        return MetricsResult(
            precision=p, recall=r, f1=f,
            class_precision=per_p, class_recall=per_r, class_f1=per_f,
            ratio_masked=ratio_masked,
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

    def get_counts(self) -> torch.Tensor:
        """Returns [num_steps] count of samples per timestep in this epoch."""
        return self.count.detach().clone()


@dataclass
class PerTimestepMetricsResult:
    """Precision, recall, F1 per diffusion timestep (averaged over S/R/O)."""

    precision: torch.Tensor  # [T]
    recall: torch.Tensor  # [T]
    f1: torch.Tensor  # [T]
    ratio_masked: torch.Tensor | None = None  # [T], fraction of valid tokens in mask state


@dataclass
class ValidationResult:
    """Result of validate(): final CARB metrics and optional per-timestep metrics."""

    carb: MetricsResult
    per_t_carb: PerTimestepMetricsResult | None = None


class PerTimestepTripletMetrics(Metric):
    """CaRB-style P/R/F1 per diffusion timestep for intermediate generation."""

    full_state_update = False

    def __init__(
        self,
        num_steps: int,
        num_classes: int = 4,
        mask_state_id: int | None = None,
    ):
        super().__init__()
        self.num_steps = num_steps
        self._per_t = [
            TripletMetrics(num_classes=num_classes, mask_state_id=mask_state_id)
            for _ in range(num_steps)
        ]

    def to(self, device):
        for m in self._per_t:
            m.to(device)
        return super().to(device)

    def update(
        self,
        intermediates: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ):
        """
        intermediates: [B, L, T] predictions at each reverse step.
        target: [B, L], mask: [B, L].
        """
        T = intermediates.shape[2]
        for t in range(T):
            self._per_t[t].update(intermediates[:, :, t], target, mask)

    def compute(self) -> PerTimestepMetricsResult:
        precisions, recalls, f1s = [], [], []
        ratio_masked_list = []
        for m in self._per_t:
            r = m.compute()
            precisions.append(r.precision)
            recalls.append(r.recall)
            f1s.append(r.f1)
            ratio_masked_list.append(r.ratio_masked)
        ratio_masked = None
        if all(x is not None for x in ratio_masked_list):
            ratio_masked = torch.tensor(ratio_masked_list, dtype=torch.float32)
        return PerTimestepMetricsResult(
            precision=torch.tensor(precisions, dtype=torch.float32),
            recall=torch.tensor(recalls, dtype=torch.float32),
            f1=torch.tensor(f1s, dtype=torch.float32),
            ratio_masked=ratio_masked,
        )

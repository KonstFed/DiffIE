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
    p = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    r = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f = 2 * p * r / (p + r) if (p + r) > 0 else float("nan")
    return p, r, f


def _metrics_single_sequence(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None,
    num_classes: int,
    mask_state_id: int | None,
) -> tuple[float, float, float, float | None]:
    """Compute P/R/F1 and ratio_masked for one sequence. pred/target/mask: [L]."""
    valid = (
        mask.bool()
        if mask is not None
        else torch.ones_like(pred, dtype=torch.bool, device=pred.device)
    )
    ratio_masked = None
    if mask_state_id is not None:
        valid_count = valid.sum().item()
        masked_count = ((pred == mask_state_id) & valid).sum().item()
        ratio_masked = masked_count / max(valid_count, 1)
    pred = pred[valid]
    target = target[valid]
    tp_list, fp_list, fn_list = [], [], []
    for c in range(num_classes):
        pc, tc = pred == c, target == c
        tp_list.append((pc & tc).sum().item())
        fp_list.append((pc & ~tc).sum().item())
        fn_list.append((~pc & tc).sum().item())
    tp = sum(tp_list[1:])
    fp = sum(fp_list[1:])
    fn = sum(fn_list[1:])
    p, r, f = _prf(tp, fp, fn)
    return p, r, f, ratio_masked


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
            "tp",
            default=torch.zeros(num_classes, dtype=torch.long),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "fp",
            default=torch.zeros(num_classes, dtype=torch.long),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "fn",
            default=torch.zeros(num_classes, dtype=torch.long),
            dist_reduce_fx="sum",
        )
        if mask_state_id is not None:
            self.add_state(
                "masked_count",
                default=torch.tensor(0, dtype=torch.long),
                dist_reduce_fx="sum",
            )
            self.add_state(
                "valid_count",
                default=torch.tensor(0, dtype=torch.long),
                dist_reduce_fx="sum",
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
            precision=p,
            recall=r,
            f1=f,
            class_precision=per_p,
            class_recall=per_r,
            class_f1=per_f,
            ratio_masked=ratio_masked,
        )


class PerTimestepLoss(Metric):
    """Tracks average loss per diffusion timestep t."""

    full_state_update = False

    def __init__(self, num_steps: int):
        super().__init__()
        self.num_steps = num_steps
        self.add_state("loss_sum", default=torch.zeros(num_steps), dist_reduce_fx="sum")
        self.add_state(
            "count",
            default=torch.zeros(num_steps, dtype=torch.long),
            dist_reduce_fx="sum",
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
    ratio_masked: torch.Tensor | None = (
        None  # [T], fraction of valid tokens in mask state
    )
    # Per-sequence trajectories [N, T] for distribution-over-t plots
    per_sequence_precision: torch.Tensor | None = None
    per_sequence_recall: torch.Tensor | None = None
    per_sequence_f1: torch.Tensor | None = None
    per_sequence_ratio_masked: torch.Tensor | None = None


@dataclass
class ValidationResult:
    """Result of validate(): final CARB metrics and optional per-timestep metrics."""

    carb: MetricsResult
    per_t_carb: PerTimestepMetricsResult | None = None


class PerTimestepTripletMetrics(Metric):
    """CaRB-style P/R/F1 per diffusion timestep for intermediate generation.

    Also computes and stores per-sequence P/R/F1/ratio_masked over t for trajectory plots.
    """

    full_state_update = False

    def __init__(
        self,
        num_steps: int,
        num_classes: int = 4,
        mask_state_id: int | None = None,
    ):
        super().__init__()
        self.num_steps = num_steps
        self.num_classes = num_classes
        self.mask_state_id = mask_state_id
        self._per_t = [
            TripletMetrics(num_classes=num_classes, mask_state_id=mask_state_id)
            for _ in range(num_steps)
        ]
        # Per-sequence trajectories (not registered as state; reset in reset())
        self._per_seq_precision: list[torch.Tensor] = []
        self._per_seq_recall: list[torch.Tensor] = []
        self._per_seq_f1: list[torch.Tensor] = []
        self._per_seq_ratio_masked: list[torch.Tensor] = []

    def reset(self):
        super().reset()
        self._per_seq_precision = []
        self._per_seq_recall = []
        self._per_seq_f1 = []
        self._per_seq_ratio_masked = []

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
        B, L, T = intermediates.shape
        for t in range(T):
            self._per_t[t].update(intermediates[:, :, t], target, mask)
        # Per-sequence metrics for each (b, t)
        m = mask if mask is not None else torch.ones_like(target, device=target.device)
        for b in range(B):
            p_list, r_list, f_list, rm_list = [], [], [], []
            for t in range(T):
                p, r, f, rm = _metrics_single_sequence(
                    intermediates[b, :, t],
                    target[b],
                    m[b],
                    self.num_classes,
                    self.mask_state_id,
                )
                p_list.append(p)
                r_list.append(r)
                f_list.append(f)
                rm_list.append(rm if rm is not None else 0.0)
            self._per_seq_precision.append(
                torch.tensor(p_list, dtype=torch.float32, device=intermediates.device)
            )
            self._per_seq_recall.append(
                torch.tensor(r_list, dtype=torch.float32, device=intermediates.device)
            )
            self._per_seq_f1.append(
                torch.tensor(f_list, dtype=torch.float32, device=intermediates.device)
            )
            self._per_seq_ratio_masked.append(
                torch.tensor(rm_list, dtype=torch.float32, device=intermediates.device)
            )

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
        # Stack per-sequence trajectories [N, T]
        per_seq_p = (
            torch.stack(self._per_seq_precision).cpu()
            if self._per_seq_precision
            else None
        )
        per_seq_r = (
            torch.stack(self._per_seq_recall).cpu() if self._per_seq_recall else None
        )
        per_seq_f = torch.stack(self._per_seq_f1).cpu() if self._per_seq_f1 else None
        per_seq_rm = (
            torch.stack(self._per_seq_ratio_masked).cpu()
            if self._per_seq_ratio_masked
            else None
        )
        return PerTimestepMetricsResult(
            precision=torch.tensor(precisions, dtype=torch.float32),
            recall=torch.tensor(recalls, dtype=torch.float32),
            f1=torch.tensor(f1s, dtype=torch.float32),
            ratio_masked=ratio_masked,
            per_sequence_precision=per_seq_p,
            per_sequence_recall=per_seq_r,
            per_sequence_f1=per_seq_f,
            per_sequence_ratio_masked=per_seq_rm,
        )

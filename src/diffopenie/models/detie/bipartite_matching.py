"""Bipartite matching (Hungarian) and IoU-based cost for DetIE.

DetIE uses order-agnostic loss: match N predicted slot label-sequences to M gold
triplets (as token-level labels), then compute loss only on matched pairs.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# Label convention: 0=O, 1=subject, 2=object, 3=predicate (same as rest of codebase)


def _one_hot_labels(labels: torch.LongTensor, num_classes: int = 4) -> torch.Tensor:
    """[B, L] label indices -> [B, L, C] one-hot (float)."""
    B, L = labels.shape
    device = labels.device
    one_hot = torch.zeros(B, L, num_classes, device=device, dtype=torch.float32)
    valid = labels >= 0
    one_hot[valid] = F.one_hot(labels[valid].clamp(0, num_classes - 1), num_classes=num_classes).float()
    return one_hot


def _span_mask_from_labels(labels: torch.LongTensor, class_idx: int) -> torch.Tensor:
    """[B, L] -> [B, L] binary mask for that class."""
    return (labels == class_idx).float()


def dice_cost(
    pred_logits: torch.Tensor,
    gold_one_hot: torch.Tensor,
    attention_mask: torch.Tensor,
    smooth: float = 1e-6,
) -> torch.Tensor:
    """
    Dice (soft IoU) cost per (pred_slot, gold_slot) for bipartite matching.
    Lower is better. Cost = 1 - dice.

    pred_logits: [B, N, L, C] (raw logits)
    gold_one_hot: [B, M, L, C] (one-hot gold labels; M = num gold per batch item)
    attention_mask: [B, L] (1 = real token)

    Returns: [B, N, M] cost matrix.
    """
    B, N, L, C = pred_logits.shape
    _, M, _, _ = gold_one_hot.shape
    pred_soft = pred_logits.softmax(dim=-1)  # [B, N, L, C]
    # Mask over sequence length L; shape [B, 1, L, 1] broadcasts to [B, N, L, C]
    mask = attention_mask.unsqueeze(1).unsqueeze(-1).float()  # [B, 1, L, 1]
    pred_mass = (pred_soft * mask).sum(dim=2)   # [B, N, C]
    # gold [B, M, L, C] * mask [B, 1, L, 1] -> [B, M, L, C]; sum dim=2 (L) -> [B, M, C]
    gold_mass = (gold_one_hot * mask).sum(dim=2)
    # pred_soft [B,N,L,C] * gold [B,M,L,C] -> [B,N,M,L,C]; mask [B,1,1,L,1]; sum over L,C -> [B,N,M]
    intersection = (
        pred_soft.unsqueeze(2) * gold_one_hot.unsqueeze(1) * mask.unsqueeze(1)
    ).sum(dim=(3, 4))
    pred_sum = pred_mass.sum(dim=2, keepdim=True)  # [B, N, 1]
    gold_sum = gold_mass.sum(dim=2).unsqueeze(1)   # [B, 1, M]
    # Force [B, N, M] so broadcasting never swaps N and M (e.g. when N=20, B=32)
    pred_sum = pred_sum.expand(-1, -1, M)
    gold_sum = gold_sum.expand(-1, N, -1)
    union = pred_sum + gold_sum - intersection
    dice = (2 * intersection + smooth) / (union + smooth)
    return 1.0 - dice  # cost: lower dice -> higher cost


def iou_cost(
    pred_logits: torch.Tensor,
    gold_one_hot: torch.Tensor,
    attention_mask: torch.Tensor,
    smooth: float = 1e-6,
) -> torch.Tensor:
    """
    IoU-based cost per (pred_slot, gold_slot). Cost = 1 - IoU.

    pred_logits: [B, N, L, C]
    gold_one_hot: [B, M, L, C]
    attention_mask: [B, L]

    Returns: [B, N, M] cost matrix.
    """
    pred_soft = pred_logits.softmax(dim=-1)
    mask = attention_mask.unsqueeze(1).unsqueeze(-1).float()  # [B, 1, L, 1]
    # For IoU we treat each (b, n, m) as: intersection over union of (pred slot n, gold slot m) over L and C
    pred_flat = (pred_soft * mask).reshape(pred_soft.shape[0], pred_soft.shape[1], -1)  # [B, N, L*C]
    gold_flat = (gold_one_hot * mask.unsqueeze(1)).reshape(gold_one_hot.shape[0], gold_one_hot.shape[1], -1)
    intersection = (pred_flat.unsqueeze(3) * gold_flat.unsqueeze(2)).sum(dim=-1)  # [B, N, M]
    pred_sum = pred_flat.sum(dim=-1, keepdim=True)
    gold_sum = gold_flat.sum(dim=-1, keepdim=True).transpose(1, 2)
    union = pred_sum + gold_sum - intersection
    iou = (intersection + smooth) / (union + smooth)
    return 1.0 - iou


def _is_empty_gold(gold_labels: torch.LongTensor) -> torch.Tensor:
    """[B, M, L] -> [B, M] true if that gold slot is empty (all O or padding)."""
    return (gold_labels == 0).all(dim=-1) | (gold_labels < 0).all(dim=-1)


def hungarian_match(cost: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Solve assignment for each batch item. cost: [B, N, M].

    Returns:
        row_ind: [B, N] – for each pred slot, which gold index it is assigned to (-1 if none)
        col_ind: [B, M] – for each gold, which pred index is assigned to it (-1 if none)

    Uses scipy.optimize.linear_sum_assignment per batch item.
    """
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        raise ImportError("DetIE bipartite matching requires scipy. Install with: pip install scipy")

    B, N, M = cost.shape
    device = cost.device
    cost_cpu = cost.detach().cpu().numpy()
    # row_ind[b, n] = gold index m that pred slot n is matched to (-1 if unmatched)
    row_ind = torch.full((B, N), -1, dtype=torch.long, device=device)
    col_ind = torch.full((B, M), -1, dtype=torch.long, device=device)

    for b in range(B):
        r, c = linear_sum_assignment(cost_cpu[b])  # r=pred indices, c=gold indices
        row_ind[b, r] = torch.from_numpy(c).to(device)
        col_ind[b, c] = torch.from_numpy(r).to(device)

    return row_ind, col_ind


class DetIELoss(nn.Module):
    """
    DetIE loss: bipartite matching + cross-entropy (or dice) on matched slots.
    Unmatched predictions matched to "no triplet" get a no-object loss.
    """

    def __init__(
        self,
        num_classes: int = 4,
        matching: str = "dice",
        no_object_weight: float = 0.1,
        use_focal: bool = False,
        focal_gamma: float = 0.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.matching = matching
        self.no_object_weight = no_object_weight
        self.use_focal = use_focal
        self.focal_gamma = focal_gamma

    def forward(
        self,
        pred_logits: torch.Tensor,
        gold_labels: torch.LongTensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """
        pred_logits: [B, N, L, C]
        gold_labels: [B, M, L] – padded with 0 (O) or -1 for no triplet
        attention_mask: [B, L]

        Returns:
            loss: scalar
            aux: dict with matching stats
        """
        B, N, L, C = pred_logits.shape
        _, M, _ = gold_labels.shape
        device = pred_logits.device

        gold_one_hot = _one_hot_labels(gold_labels.view(B, M * L), C).view(B, M, L, C)

        if self.matching == "dice":
            cost = dice_cost(pred_logits, gold_one_hot, attention_mask)
        else:
            cost = iou_cost(pred_logits, gold_one_hot, attention_mask)

        # For "no triplet" gold slots, set cost to a small value so they get matched to extra predictions
        empty_gold = _is_empty_gold(gold_labels)  # [B, M]
        cost = cost.clone()
        cost[:, :, empty_gold] -= 10.0  # make empty gold very attractive for extra preds

        row_ind, col_ind = hungarian_match(cost)

        # row_ind[b, n] = m means pred slot n is matched to gold m
        # Build matched pred and gold for loss
        total_loss = 0.0
        num_matched = 0
        num_no_obj = 0

        for b in range(B):
            for n in range(N):
                m = row_ind[b, n].item()
                if m < 0:
                    continue
                gold_l = gold_labels[b, m]  # [L]
                pred_l = pred_logits[b, n]  # [L, C]
                if empty_gold[b, m].item():
                    num_no_obj += 1
                    # No-object: encourage all O (class 0)
                    loss_n = F.cross_entropy(
                        pred_l.reshape(-1, C),
                        torch.zeros(L, dtype=torch.long, device=device),
                        reduction="mean",
                    )
                    total_loss = total_loss + self.no_object_weight * loss_n
                else:
                    num_matched += 1
                    if self.use_focal and self.focal_gamma > 0:
                        pt = pred_l.softmax(dim=-1).gather(-1, gold_l.clamp(0, C - 1).unsqueeze(-1)).squeeze(-1)
                        focal = (1 - pt) ** self.focal_gamma
                        loss_n = F.cross_entropy(
                            pred_l.reshape(-1, C),
                            gold_l.clamp(0, C - 1).reshape(-1),
                            reduction="none",
                        ).reshape(L)
                        loss_n = (focal * loss_n * attention_mask[b].float()).sum() / (attention_mask[b].sum().float().clamp(min=1))
                    else:
                        loss_n = F.cross_entropy(
                            pred_l.reshape(-1, C),
                            gold_l.clamp(0, C - 1).reshape(-1),
                            reduction="none",
                        ).reshape(L)
                        loss_n = (loss_n * attention_mask[b].float()).sum() / (attention_mask[b].sum().float().clamp(min=1))
                    total_loss = total_loss + loss_n

        total_pairs = num_matched + num_no_obj
        if total_pairs > 0:
            total_loss = total_loss / total_pairs

        aux = {"num_matched": num_matched, "num_no_obj": num_no_obj}
        return total_loss, aux

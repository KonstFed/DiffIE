import torch
from torch import Tensor

Span = tuple[int, int] | None
Triplet = tuple[Span, Span, Span]  # (sub_span, obj_span, pred_span)
TokenSet = set[int] | list[int] | None


def _iou_spans(a: tuple[int, int], b: tuple[int, int]) -> float:
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]) + 1)
    union = (a[1] - a[0] + 1) + (b[1] - b[0] + 1) - inter
    return inter / union if union > 0 else 0.0


def _iou_tokens(a: set[int] | list[int], b: set[int] | list[int]) -> float:
    sa, sb = set(a), set(b)
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union > 0 else 0.0


def triplet_dist_spans(a: Triplet, b: Triplet) -> float:
    """
    Similarity between two triplets as mean span IoU.

    Accepts triplets in (sub_span, obj_span, pred_span) form — the same layout
    used throughout the codebase. Each span is an inclusive (start, end) word-level
    tuple or None. None on either side contributes 0 to the mean.

    Returns 1/3 * (IoU(sub) + IoU(obj) + IoU(pred)).
    """
    sub_a, obj_a, pred_a = a
    sub_b, obj_b, pred_b = b
    scores = [
        _iou_spans(x, y) if x is not None and y is not None else 0.0
        for x, y in ((sub_a, sub_b), (obj_a, obj_b), (pred_a, pred_b))
    ]
    return sum(scores) / 3


def triplet_dist_tokens(
    sub_a: TokenSet, obj_a: TokenSet, pred_a: TokenSet,
    sub_b: TokenSet, obj_b: TokenSet, pred_b: TokenSet,
) -> float:
    """
    Similarity between two triplets as mean token-set IoU.

    Each argument is a collection of token indices (set or list) or None.
    None on either side contributes 0 to the mean.

    Returns 1/3 * (IoU(sub) + IoU(obj) + IoU(pred)).
    """
    scores = [
        _iou_tokens(x, y) if x is not None and y is not None else 0.0
        for x, y in ((sub_a, sub_b), (obj_a, obj_b), (pred_a, pred_b))
    ]
    return sum(scores) / 3


def per_token_entropy(
    samples: Tensor,
    num_classes: int = 4,
    normalize: bool = True,
    miller_madow: bool = True,
) -> Tensor:
    """Compute per-token entropy from K diffusion samples.

    Args:
        samples: (K, L) integer tensor of predicted labels per sample.
        num_classes: number of label classes (default 4: B, S, R, O).
        normalize: if True, divide by log(num_classes) to get [0, 1] range.
        miller_madow: if True, apply Miller-Madow bias correction.

    Returns:
        (L,) tensor of per-token entropy values.
    """
    K, L = samples.shape

    # empirical counts: (L, num_classes)
    counts = torch.zeros(L, num_classes, device=samples.device)
    counts.scatter_add_(1, samples.T, torch.ones_like(samples.T, dtype=torch.float))

    # empirical distribution
    p = counts / K  # (L, num_classes)

    # entropy: -sum p log p, with 0 log 0 = 0
    log_p = torch.where(p > 0, torch.log(p), torch.zeros_like(p))
    H = -(p * log_p).sum(dim=1)  # (L,)

    if miller_madow:
        num_nonzero = (counts > 0).sum(dim=1).float()  # (L,)
        H = H + (num_nonzero - 1) / (2 * K)

    if normalize:
        H = H / torch.log(torch.tensor(num_classes, dtype=H.dtype))

    return H


def sentence_entropy_stats(H: Tensor, tau: float = 0.3) -> dict[str, float]:
    """Aggregate per-token entropy into sentence-level summaries."""
    return {
        "mean": H.mean().item(),
        "max": H.max().item(),
        "frac_unstable": (H > tau).float().mean().item(),
    }


def longest_true_run(x: torch.Tensor) -> tuple[int, int] | None:
    """
    Find the longest run of True values in a 1D boolean tensor.
    """
    x = x.to(torch.int32)

    padded = torch.cat([torch.tensor([0]), x, torch.tensor([0])])

    # find run boundaries
    diff = padded[1:] - padded[:-1]

    starts = torch.where(diff == 1)[0]
    ends = torch.where(diff == -1)[0] - 1

    if len(starts) == 0:
        return None  # no True values

    lengths = ends - starts + 1
    i = torch.argmax(lengths)

    return int(starts[i]), int(ends[i])


def extract_longest_span(pred_indices: torch.BoolTensor, word_ids: list[int | None]) -> tuple[int, int] | None:
    """
    Extract the longest span, which will not split any word.
    """
    word_starts = []
    prev = None

    for i, w in enumerate(word_ids):
        if w is not None and w != prev:
            word_starts.append(i)
        prev = w

    per_word_pred = []
    for i in range(len(word_starts)):
        start = word_starts[i]
        if i == len(word_starts) - 1:
            end = len(pred_indices)
        else:
            end = word_starts[i+1]
        per_word_pred.append(pred_indices[start:end].all())
    return longest_true_run(torch.tensor(per_word_pred))

import torch
from torch import Tensor


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

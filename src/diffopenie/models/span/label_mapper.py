"""Simplex mapper based on https://arxiv.org/pdf/2309.02530"""
from abc import ABC, abstractmethod
import torch


class BaseMapper(ABC):
    @abstractmethod
    def get_random(
        self,
        n: int,
        sentence_len: torch.LongTensor,
        device: torch.device | None = None,
    ) -> torch.FloatTensor:
        """Get random x_t for inference,

        Args:
            n (int): number of samples
            sentence_len (torch.LongTensor): tensor of sentence lengths with size [n]

        Returns:
            torch.FloatTensor: x_t
        """

    def forward(self, labels: torch.LongTensor, sentence_len: torch.LongTensor) -> torch.FloatTensor:
        """Map triplet into continuous (logit) space with label smoothing.

        Args:
            labels (torch.LongTensor): spans with size [n, 6]
            sentence_len (torch.LongTensor): tensor of sentence lengths with size [n]

        Returns:
            torch.FloatTensor: x_t
        """

    def reverse(self, x: torch.FloatTensor, sentence_len: torch.LongTensor) -> torch.LongTensor:
        """Reverse from continuous to discrete spans.

        Args:
            x (torch.FloatTensor): [n, ...] continuous vector of n samples
            sentence_len (torch.LongTensor): sentence lengths for all samples with size [n]

        Returns:
            torch.LongTensor: [n, 6] triplets spans
        """

class FloatIndexMapper(BaseMapper):
    """Map triplets in DiffusionNER style, with index normalization with sentence length.
    """
    def get_random(self, n: int, sentence_len: torch.LongTensor) -> torch.FloatTensor:
        """Get random x_t for inference.

        Args:
            n (int): number of samples
            sentence_len (torch.LongTensor): tensor of sentence lengths with size [n]

        Returns:
            torch.FloatTensor: of size [n, 6]
        """
        # maybe normal?
        # or sample only correct spans?
        return torch.rand(n, 6)

    def forward(self, labels: torch.LongTensor, sentence_len: torch.LongTensor) -> torch.FloatTensor:
        """Map triplets into continuous index space

        Args:
            labels (torch.LongTensor): of size [n, 6] of triple (s_l, s_r, o_l, o_r, p_l, p_r)
            sentence_len (torch.LongTensor): of size [n]

        Returns:
            torch.FloatTensor: of size [n, 6]
        """
        return labels / sentence_len.unsqueeze(1)

    def reverse(self, x: torch.FloatTensor, sentence_len: torch.LongTensor) -> torch.LongTensor:
        """Reverse from continuous index space to discrete spans.

        Args:
            x (torch.FloatTensor): of size [n, 6]
            sentence_len (torch.LongTensor): of size [n]

        Returns:
            torch.LongTensor: of size [n, 6]
        """
        labels = x * sentence_len.unsqueeze(1)
        labels = labels.round().int()
        labels = labels.clamp(torch.zeros_like(labels), sentence_len.unsqueeze(1) - 1)
        return labels


class ContinuousSpanMapper(BaseMapper):
    """Map triplets into continuous space.

    Label smoothing (as in the paper): given one-hot x_0 on the simplex boundary,
    we use x̃_0 = (1-ε)*x_0 + ε*(1/d) so that x̃_0 is in the interior (strict positivity).
    This makes the inverse additive-logistic transform to logit space well-defined
    and avoids infinite log-odds at simplex vertices.
    """

    def __init__(self, logit_value: float = 1.0, epsilon: float = 0.01):
        self.logit_value = logit_value
        self.epsilon = epsilon  # label smoothing: x̃_0 = (1-ε)*x_0 + ε/d

    def get_random(
        self,
        n: int,
        sentence_len: torch.LongTensor,
    ) -> torch.FloatTensor:
        """Random x_t for inference. sentence_len [B]; output [n, 6, L_max-1]."""
        L_max = max(int(sentence_len.max().item()), 2)
        return torch.randn(n, 6, L_max - 1)

    @staticmethod
    def logits_to_probs(
        y: torch.FloatTensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.FloatTensor:
        """Inverse additive-logistic: y [..., d-1] -> x [..., d] on the simplex.

        Equivalent to softmax(concat(y, 0)): last coordinate is reference (log 1 = 0).
        If attention_mask is given (True = valid, False = pad), pad positions in y
        get float('-inf') before concat, so the appended 0 is never masked.
        """
        if attention_mask is not None:
            # mask over first d-1 positions only; last (reference) stays 0
            mask_first = attention_mask[..., :-1].bool()  # [..., d-1], True = valid
            pad_mask = ~mask_first
            while pad_mask.dim() < y.dim():
                pad_mask = pad_mask.unsqueeze(-2)
            y = y.masked_fill(pad_mask, float("-inf"))
        z = torch.cat([torch.zeros_like(y[..., :1]), y], dim=-1)
        return torch.softmax(z, dim=-1)

    def forward_index(
        self,
        index: torch.LongTensor,
        sentence_len: torch.LongTensor,
    ) -> torch.FloatTensor:
        """Map index into continuous (logit) space with label smoothing.

        sentence_len [B]: d_b = number of valid positions per example.
        One-hot x_0 -> x̃_0 = (1-ε)*x_0 + ε/d_b (only over valid), then
        y_0_i = log(x̃_0_i / x̃_0_0) with first valid = reference (matches logits_to_probs).
        Returns [B, L_max - 1]; invalid positions zeroed.
        """
        B = index.shape[0]
        device = index.device
        L_max = max(int(sentence_len.max().item()), 2)

        one_hot = torch.zeros(B, L_max, device=device, dtype=torch.float32)
        idx = index.clamp(0, L_max - 1).squeeze(1)
        one_hot.scatter_(1, idx.unsqueeze(1), 1.0)

        d_expanded = sentence_len.unsqueeze(1).float().clamp(min=1)
        smooth = (1.0 - self.epsilon) * one_hot + self.epsilon / d_expanded
        valid_mask = (
            torch.arange(L_max, device=device).unsqueeze(0)
            < sentence_len.unsqueeze(1)
        )
        smooth = smooth * valid_mask.float()
        # Sanity: sum over valid positions should be 1
        # row_sum = smooth.sum(dim=-1)
        # max_err = (row_sum - 1.0).abs().max().item()
        # assert max_err < 1e-5, f"smooth sum ~1, |sum-1|_max={max_err:.2e}"

        # First position as reference (same as logits_to_probs: concat(0, y) then softmax)
        ref = smooth[:, 0:1].clamp(min=1e-9)
        out = torch.log(smooth[:, 1:] / ref)
        out_valid = (
            torch.arange(L_max - 1, device=device).unsqueeze(0)
            < (sentence_len - 1).unsqueeze(1)
        )
        out[~out_valid] = 0.0
        return out * out_valid.float()

    def reverse(
        self,
        logits: torch.FloatTensor,
        sentence_len: torch.LongTensor,
    ) -> torch.LongTensor:
        """Logits [..., d-1] -> probs [..., d], then argmax.

        sentence_len [B]: mask positions >= sentence_len[b] so argmax is valid.
        """
        probs = self.logits_to_probs(logits)
        L_max = probs.shape[-1]
        invalid = (
            torch.arange(L_max, device=sentence_len.device).unsqueeze(0)
            >= sentence_len.unsqueeze(1)
        )
        if probs.dim() == 3:
            invalid = invalid.unsqueeze(1).expand(-1, 6, -1)
        probs = probs.masked_fill(invalid, -float("inf"))
        return torch.argmax(probs, dim=-1).long()


    def forward(
        self,
        labels: torch.LongTensor,
        sentence_len: torch.LongTensor,
    ) -> torch.FloatTensor:
        """Map triplet labels into continuous space.

        labels [B, 6]; sentence_len [B] per-example length.
        Returns flattened [B, 6*(L_max-1)]; reshape to [B, 6, L_max-1].
        """
        assert labels.dim() == 2
        B, n_elems = labels.shape
        flat = labels.reshape(-1, 1)
        len_flat = sentence_len.unsqueeze(1).expand(-1, n_elems).reshape(-1)
        dist = self.forward_index(flat, len_flat)
        return dist.reshape(B, n_elems * dist.shape[1])



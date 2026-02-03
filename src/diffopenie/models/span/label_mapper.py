import torch
from torch import nn
from pydantic import BaseModel

class ContinuousSpanMapper:
    """Map triplets into continuous space"""

    def __init__(self, logit_value: float = 1.0):
        self.logit_value = logit_value

    def forward_index(
        self, index: torch.LongTensor, sentence_len: int,
    ) -> torch.FloatTensor:
        """Map index into continuous space.

        Args:
            index (torch.LongTensor): Shape [B, 1]; index[b, 0] is the position.
            sentence_len (int): Length of the sentence.

        Returns:
            torch.FloatTensor: Shape [B, sentence_len]; zeros with logit_value
                at index[b, 0] for each b.
        """
        B = index.shape[0]
        out = torch.zeros(
            B, sentence_len,
            device=index.device,
            dtype=torch.float32,
        )
        value = torch.full(
            (B, 1), self.logit_value,
            device=index.device, dtype=torch.float32,
        )
        out.scatter_(1, index, value)
        return out

    def reverse_index(self, logits: torch.FloatTensor) -> torch.LongTensor:
        """Reverse the mapping from continuous space to index."""
        return torch.argmax(logits, dim=-1)

    def forward(
        self,
        labels: torch.LongTensor,
        sentence_len: int,
    ) -> torch.FloatTensor:
        """Map triplet labels into continuous space.

        Each label vector should have 6 elements: [S_l, S_r
        where:
        - S_l: Start index of the subject span
        - S_r: End index of the subject span
        - O_l: Start index of the object span
        - O_r: End index of the object span
        - P_l: Start index of the predicate span
        - P_r: End index of the predicate span

        Args:
            labels (torch.LongTensor): Label vector [B, 6]
            sentence_len (int): Length of the sentence.

        Returns:
            torch.FloatTensor: Continuous space vector [B, 6, sentence_len]
        """
        assert labels.dim() == 2
        B, n_elems = labels.shape
        flat = labels.reshape(-1, 1)  # [B*n_elems]
        dist = self.forward_index(flat, sentence_len)  # [B*n_elems, sentence_len]
        return dist.reshape(B, n_elems * sentence_len)  # [B, n_elems * sentence_len]

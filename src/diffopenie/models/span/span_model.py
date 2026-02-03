from torch import nn
import torch

from diffopenie.models.base_model import BaseTripletModel
from diffopenie.models.encoder import BERTEncoder
from diffopenie.models.span import ContinuousSpanMapper
from diffopenie.models.span.denoiser import SpanDenoiser
from diffopenie.diffusion.scheduler import LinearScheduler


class SpanDiffusionModel(nn.Module, BaseTripletModel):
    """
    Unified diffusion model for span labeling.
    """

    def __init__(
        self,
        denoiser: SpanDenoiser,
        scheduler: LinearScheduler,
        label_mapper: ContinuousSpanMapper,
        encoder: BERTEncoder,
    ):
        self.denoiser = denoiser
        self.scheduler = scheduler
        self.label_mapper = label_mapper
        self.encoder = encoder

    def train_step(
        self,
        token_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        label_spans: torch.LongTensor,
    ) -> torch.LongTensor:
        """Train step for the model.

        Args:
            token_ids: Token IDs [B, L]
            attention_mask: Attention mask [B, L]
            label_spans: Label spans [B, 6]

        Returns:
            torch.LongTensor: _description_
        """
        token_embeddings = self.encoder(token_ids, attention_mask)
        return self.label_mapper(token_embeddings)

    def noise(self, x_0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """a.k.a. forward process

        x_0 - [B, 6, L]"""

        # apply normal noise as usual
        noise = torch.randn_like(x_0)
        x_t = self.scheduler.q_sample(x_0, t, noise)

        # Project to zero-mean since softmax is invariant to adding a constant
        # (i.e. shifts in the all-ones direction do not change probabilities)
        x_t = x_t - x_t.mean(dim=-1, keepdim=True)

    def denoise(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """a.k.a. reverse process"""
        # TODO: maybe in future apply some tricks over denoised probabilities
        return self.denoiser.forward(
            x_t=x_t,
            t=t,
            token_embeddings=token_embeddings,
            attn_mask=attention_mask,
        )

    def encode_tokens(
        self,
        token_ids: torch.LongTensor,  # [B, L]
        attention_mask: torch.Tensor,  # [B, L]
    ) -> torch.Tensor:
        """
        Encode tokens using the BERT encoder.

        Args:
            token_ids: Token IDs [B, L]
            attention_mask: Attention mask [B, L]

        Returns:
            Token embeddings [B, L, bert_dim]
        """
        return self.encoder(token_ids, attention_mask)

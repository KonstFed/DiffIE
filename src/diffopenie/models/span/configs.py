from pydantic import BaseModel
from diffopenie.models.encoder import BERTEncoderConfig
from diffopenie.models.span.label_mapper import ContinuousSpanMapper
from diffopenie.diffusion.scheduler import LinearSchedulerConfig
from diffopenie.models.span.span_model import SpanDiffusionModel
from diffopenie.models.span.slot_denoiser import SlotDenoiser




class SlotDenoiserConfig(BaseModel):
    """Config for SpanDenoiser (slot decoder + pointer)."""

    bert_dim: int = 768
    num_steps: int = 1000
    d_model: int | None = None
    n_heads: int = 8
    n_layers: int = 1
    d_ff: int | None = None
    dropout: float = 0.1

    def create(self) -> SlotDenoiser:
        return SlotDenoiser(
            bert_dim=self.bert_dim,
            num_steps=self.num_steps,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            d_ff=self.d_ff,
            dropout=self.dropout,
        )

DenoiserConfig = SlotDenoiserConfig


class ContinuousSpanMapperConfig(BaseModel):
    """Config for ContinuousSpanMapper (span indices -> logits)."""

    logit_value: float = 1.0
    epsilon: float = 0.01  # label smoothing: x̃_0 = (1-ε)*x_0 + ε/d

    def create(self) -> ContinuousSpanMapper:
        return ContinuousSpanMapper(
            logit_value=self.logit_value,
            epsilon=self.epsilon,
        )

class SpanDiffusionModelConfig(BaseModel):
    """Config for SpanDiffusionModel (encoder + label_mapper + scheduler + denoiser)."""

    encoder: BERTEncoderConfig | None = None
    label_mapper: ContinuousSpanMapperConfig
    scheduler: LinearSchedulerConfig
    denoiser: DenoiserConfig

    def create(self) -> SpanDiffusionModel:
        encoder_instance = self.encoder.create() if self.encoder is not None else None
        label_mapper_instance = self.label_mapper.create()
        scheduler_instance = self.scheduler.create()
        denoiser_instance = self.denoiser.create()
        return SpanDiffusionModel(
            denoiser=denoiser_instance,
            scheduler=scheduler_instance,
            label_mapper=label_mapper_instance,
            encoder=encoder_instance,
        )

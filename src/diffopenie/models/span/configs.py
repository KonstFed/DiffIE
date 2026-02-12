from typing import Annotated, Literal

from pydantic import BaseModel, Field

from diffopenie.models.encoder import BERTEncoderConfig
from diffopenie.models.span.label_mapper import ContinuousSpanMapper, FloatIndexMapper
from diffopenie.diffusion.scheduler import LinearSchedulerConfig
from diffopenie.models.span.span_model import SpanDiffusionModel
from diffopenie.models.span.slot_denoiser import SlotDenoiser
from diffopenie.models.span.diffusionNER import DiffusionNERDenoiser


# mappers


class FloatIndexMapperConfig(BaseModel):
    """Config for FloatIndexMapper (span indices -> float indices)."""

    type: Literal["float_index"] = "float_index"

    def create(self) -> FloatIndexMapper:
        return FloatIndexMapper()


class ContinuousSpanMapperConfig(BaseModel):
    """Config for ContinuousSpanMapper (span indices -> logits)."""

    type: Literal["continuous"] = "continuous"

    logit_value: float = 1.0
    epsilon: float = 0.01  # label smoothing: x̃_0 = (1-ε)*x_0 + ε/d

    def create(self) -> ContinuousSpanMapper:
        return ContinuousSpanMapper(
            logit_value=self.logit_value,
            epsilon=self.epsilon,
        )


LabelMapperConfig = FloatIndexMapperConfig | ContinuousSpanMapperConfig


class DiffusionNERDenoiserConfig(BaseModel):
    """Config for DiffusionNERDenoiser (diffusion NER denoiser)."""

    type: Literal["diffusion_ner"] = "diffusion_ner"
    label_mapper: FloatIndexMapperConfig
    embedder_dim: int = 768
    span_dim: int = 128
    num_steps: int = 1000
    cross_attn_heads: int = 8
    cross_attn_layers: int = 1
    cross_attn_dropout: float = 0.1
    self_attn_heads: int = 8
    self_attn_layers: int = 1
    self_attn_dropout: float = 0.1

    def create(self) -> DiffusionNERDenoiser:
        return DiffusionNERDenoiser(
            label_mapper=self.label_mapper.create(),
            embedder_dim=self.embedder_dim,
            num_steps=self.num_steps,
            cross_attn_heads=self.cross_attn_heads,
            cross_attn_layers=self.cross_attn_layers,
            cross_attn_dropout=self.cross_attn_dropout,
            self_attn_heads=self.self_attn_heads,
            self_attn_layers=self.self_attn_layers,
            self_attn_dropout=self.self_attn_dropout,
        )


class SlotDenoiserConfig(BaseModel):
    """Config for SpanDenoiser (slot decoder + pointer)."""

    type: Literal["slot"] = "slot"
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


DenoiserConfig = Annotated[
    SlotDenoiserConfig | DiffusionNERDenoiserConfig,
    Field(discriminator="type"),
]

# model


class SpanDiffusionModelConfig(BaseModel):
    """Config for SpanDiffusionModel (encoder + label_mapper + scheduler + denoiser)."""

    encoder: BERTEncoderConfig | None = None
    label_mapper: LabelMapperConfig
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

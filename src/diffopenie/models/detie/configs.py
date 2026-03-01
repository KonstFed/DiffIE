"""Pydantic configs for DetIE model."""

from typing import Literal

from pydantic import BaseModel, ConfigDict

from diffopenie.models.encoder import BERTEncoderConfig
from diffopenie.models.detie.detie_model import DetIEModel


class DetIEModelConfig(BaseModel):
    """Config for DetIE model (encoder + slot decoder + bipartite matching)."""

    model_config = ConfigDict(extra="forbid")
    type: Literal["detie"] = "detie"
    encoder: BERTEncoderConfig
    num_slots: int = 20
    d_model: int = 512
    n_heads: int = 8
    n_decoder_layers: int = 2
    d_ff: int | None = None
    dropout: float = 0.1
    num_classes: int = 4

    def create(self) -> DetIEModel:
        encoder = self.encoder.create()
        return DetIEModel(
            encoder=encoder,
            num_slots=self.num_slots,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_decoder_layers=self.n_decoder_layers,
            d_ff=self.d_ff,
            dropout=self.dropout,
            num_classes=self.num_classes,
        )

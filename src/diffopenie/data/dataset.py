import os
import logging
from typing import Annotated, Literal, Union

import torch
from pydantic import BaseModel, Field
from torch.utils.data import Dataset
from tqdm import trange

from diffopenie.data.imojie import SequenceImojieDatasetConfig
from diffopenie.data.lsoie import (
    SequenceLSOEIDataset,
    SequenceLSOEIDatasetConfig,
    SpanLSOIEDataset,
    SpanLSOEIDatasetConfig,
)
from diffopenie.models.encoder import BERTEncoder, BERTEncoderConfig

__all__ = [
    "CachedDataset",
    "CachedDatasetConfig",
    "SequenceLSOEIDataset",
    "SequenceLSOEIDatasetConfig",
    "SpanLSOIEDataset",
    "SpanLSOEIDatasetConfig",
]

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)


def _pad_token_ids(
    token_ids_list: list[list[int]],
    pad_token_id: int = 0,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad token_ids to same length; return (padded_ids, attention_mask)."""
    max_len = max(len(ids) for ids in token_ids_list)
    pad_id = pad_token_id if pad_token_id is not None else 0
    device = device or torch.device("cpu")
    padded = []
    mask = []
    for ids in token_ids_list:
        L = len(ids)
        padded.append(ids + [pad_id] * (max_len - L))
        mask.append([1] * L + [0] * (max_len - L))
    return (
        torch.tensor(padded, dtype=torch.long, device=device),
        torch.tensor(mask, dtype=torch.long, device=device),
    )


class CachedDataset(Dataset):
    """Wrap datasets; optionally run BERT encoder and cache token_embeddings."""

    def __init__(
        self,
        datasets: list[Dataset],
        encoder: BERTEncoder | None = None,
        batch_size: int = 32,
    ):
        self._cache: list[dict] = []
        items: list[dict] = []
        for ds in datasets:
            for i in range(len(ds)):
                items.append(ds[i])

        if encoder is None:
            self._cache = items
        else:
            self._encode_and_cache(items, encoder, batch_size)

    def _encode_and_cache(
        self, items: list[dict], encoder: BERTEncoder, batch_size: int
    ) -> None:
        logger.info("Precomputing token embeddings in CachedDataset")
        encoder.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoder.to(device)
        pad_token_id = getattr(
            encoder.tokenizer, "pad_token_id", None
        ) or encoder.tokenizer.cls_token_id

        with torch.no_grad():
            for start in trange(0, len(items), batch_size):
                end = min(start + batch_size, len(items))
                batch_items = items[start:end]
                token_ids_list = [
                    item["token_ids"]
                    if isinstance(item["token_ids"], list)
                    else item["token_ids"].tolist()
                    for item in batch_items
                ]
                input_ids, attention_mask = _pad_token_ids(
                    token_ids_list, pad_token_id=pad_token_id, device=device
                )
                batch_embeddings = encoder.forward(input_ids, attention_mask)
                for j, item in enumerate(batch_items):
                    mask = attention_mask[j] == 1
                    arr = batch_embeddings[j, mask].cpu().numpy()
                    item = dict(item)
                    item["token_embeddings"] = arr
                    self._cache.append(item)

        encoder.to("cpu")

    def __len__(self) -> int:
        return len(self._cache)

    def __getitem__(self, idx: int) -> dict:
        return self._cache[idx]


DatasetConfig = Annotated[
    Union[
        SpanLSOEIDatasetConfig,
        SequenceLSOEIDatasetConfig,
        SequenceImojieDatasetConfig,
    ],
    Field(discriminator="type"),
]


class CachedDatasetConfig(BaseModel):
    """Configuration for CachedDataset: list of LSOIE dataset configs + encoder."""

    type: Literal["cached"] = "cached"
    datasets: list[DatasetConfig]
    encoder: BERTEncoderConfig | None = None
    batch_size: int = 32

    def create(self, split: str | list[str]) -> CachedDataset:
        built = [cfg.create(split) for cfg in self.datasets]
        encoder_instance = (
            self.encoder.create() if self.encoder is not None else None
        )
        return CachedDataset(
            built,
            encoder=encoder_instance,
            batch_size=self.batch_size,
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s [%(name)s] %(message)s",
    )
    ds = SpanLSOIEDataset(split="train")
    print(ds[0])

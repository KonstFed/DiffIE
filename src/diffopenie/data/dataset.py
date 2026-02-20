import os
import logging

import pandas as pd
import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import Literal
from pydantic import BaseModel
from tqdm import trange

from diffopenie.models.encoder import BERTEncoder, BERTEncoderConfig

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)


def get_left_right_border(label: list[str], tag_prefix: str) -> tuple[int, int]:
    """
    Returns the (left, right) span indices for the specified tag prefix within a sequence of labels.

    Examples:
        example = ["O", "A0-B", "A0-I", "P-B", "P-I", "O", "A1-B", "A1-I"]

        get_left_right_border(example, "A0")  # returns (1, 2)
        get_left_right_border(example, "A1")  # returns (6, 7)
        get_left_right_border(example, "P")   # returns (3, 4)
    """
    try:
        start = label.index(f"{tag_prefix}-B")
        end = start
        for i in range(start + 1, len(label)):
            if label[i] != f"{tag_prefix}-I":
                break
            end = i
        return start, end
    except ValueError:
        # Tag not found, return None indices
        return None, None


def labels_to_indices(
    label: list[str],
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    """Convert labels to word-level indices for A0 (subject), A1 (object), and P (predicate)."""
    # TODO: extract_longest_span in future if will be used
    return (
        get_left_right_border(label, "A0"),
        get_left_right_border(label, "A1"),
        get_left_right_border(label, "P"),
    )


def word_to_token_indices(
    word_ids: list[int | None], word_start: int, word_end: int
) -> tuple[int | None, int | None]:
    """Convert word indices to token indices."""
    if word_start is None or word_end is None:
        return None, None

    token_start = next((i for i, wid in enumerate(word_ids) if wid == word_start), None)
    token_end = next(
        (i for i in range(len(word_ids) - 1, -1, -1) if word_ids[i] == word_end), None
    )
    return token_start, token_end


def _is_label_continous(labels: list[str], prefix: str) -> bool:
    """Check if given label is continous"""
    first_index = next((i for i, l in enumerate(labels) if l.startswith(prefix)), None)
    last_index = next(
        (i for i in range(len(labels) - 1, -1, -1) if labels[i].startswith(prefix)),
        None,
    )
    if first_index is None or last_index is None:
        return False
    return all(lab.startswith(prefix) for lab in labels[first_index : last_index + 1])


class CachedDataset(Dataset):
    """Dataset that caches the data in memory."""

    def __init__(self, dataset: Dataset):
        self._cache = []
        for i in range(len(dataset)):
            self._cache.append(dataset[i])

    def __len__(self) -> int:
        return len(self._cache)

    def __getitem__(self, idx: int) -> dict:
        return self._cache[idx]


class SpanLSOIEDataset:
    """Dataset for the LSOIE dataset."""

    # TODO: add option to compute Bert Embeddings on init
    # to increase performance

    def __init__(
        self,
        split: str | list[str] = "train",
        tokenizer_name: str = "bert-base-uncased",
        filter_spans: bool = True,
        encoder: BERTEncoder | None = None,
    ):
        splits = [split] if isinstance(split, str) else split
        self.split = splits

        hf_dataset = load_dataset("wardenga/lsoie", trust_remote_code=True)
        dfs = [pd.DataFrame(hf_dataset[s]) for s in splits]
        dataset = pd.concat(dfs, ignore_index=True).reset_index(drop=True)
        dataset["sentence"] = dataset["words"].apply(lambda x: " ".join(x))
        self.dataset = dataset.sort_values(by="sentence").reset_index(drop=True)

        self.filter_spans = filter_spans
        if filter_spans:
            self._filter()
        else:
            logger.info("Span dataset filtering: disabled")


        self.encoder = encoder
        if encoder is None:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        else:
            self.tokenizer = encoder.tokenizer
            self._encode_tokens()


    def _encode_tokens(self, batch_size: int = 32) -> None:
        logger.info("Precomputing token embeddings")
        # ugly precomputing of token embeddings
        self.encoder.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder.to(device)
        n = len(self.dataset)
        # Pre-create column so each cell can hold a 2D array (object dtype)
        self.dataset["token_embeddings"] = [None] * n
        col_idx = self.dataset.columns.get_loc("token_embeddings")
        with torch.no_grad():
            for start in trange(0, n, batch_size):
                end = min(start + batch_size, n)
                batch_indices = list(range(start, end))
                batch_words = [
                    self.get_words_item(i)["words"] for i in batch_indices
                ]
                encoding = self.tokenizer(
                    batch_words,
                    is_split_into_words=True,
                    add_special_tokens=False,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                input_ids = encoding["input_ids"].to(device)
                attention_mask = encoding["attention_mask"].to(device)
                batch_embeddings = self.encoder.forward(input_ids, attention_mask)
                for j, i in enumerate(batch_indices):
                    mask = attention_mask[j] == 1
                    arr = batch_embeddings[j, mask].cpu().numpy()
                    self.dataset.iat[i, col_idx] = arr

        self.encoder.to("cpu")
        del self.encoder
        self.encoder = True

    def __len__(self) -> int:
        return len(self.dataset)

    def _filter(self) -> None:
        """Filter triplets with None or not continous spans"""
        to_drop: list[int] = []
        none_idx: list[int] = []
        for i in range(len(self.dataset)):
            row = self.get_words_item(i)
            labels = row["labels"]
            if (
                row["subject_span"] is None
                or row["object_span"] is None
                or row["relation_span"] is None
            ):
                none_idx.append(i)
                continue

            if (
                not _is_label_continous(labels, "A0")
                or not _is_label_continous(labels, "A1")
                or not _is_label_continous(labels, "P")
            ):
                to_drop.append(i)

        total_filtered = len(none_idx) + len(to_drop)
        logger.info(
            "Span dataset filtering: total_filtered=%d (None=%d, non_continuous=%d)",
            total_filtered,
            len(none_idx),
            len(to_drop),
        )
        self.dataset = self.dataset.drop(to_drop + none_idx)

    def get_words_item(
        self,
        idx: int,
    ) -> dict[str, list]:
        row = self.dataset.iloc[idx]
        words = row["words"]
        labels = row["label"]

        # Get word-level indices for A0, A1, P
        (s_l, s_r), (o_l, o_r), (p_l, p_r) = labels_to_indices(labels)
        return {
            "words": words,
            "labels": labels,
            # if any is None, span doesn't exists
            "subject_span": (s_l, s_r) if s_l is not None else None,
            "object_span": (o_l, o_r) if o_l is not None else None,
            "relation_span": (p_l, p_r) if p_l is not None else None,
        }

    def __getitem__(self, idx: int) -> dict:
        """get tokens and triple for the given index

        Args:
            idx (int)

        Returns:
            dict: tokens, token_ids, spans
        """
        row = self.get_words_item(idx)
        words = row["words"]
        # ugly but ok
        token_embeddings = self.dataset.iloc[idx]["token_embeddings"] if self.encoder is not None else None
        s_l, s_r = row["subject_span"]
        o_l, o_r = row["object_span"]
        p_l, p_r = row["relation_span"]

        # Tokenize words
        encoding = self.tokenizer(
            words, is_split_into_words=True, add_special_tokens=False
        )

        tokens = self.tokenizer.convert_ids_to_tokens(encoding["input_ids"])
        word_ids = encoding.word_ids()


        # Convert word indices to token indices
        token_triplets = (
            word_to_token_indices(word_ids, s_l, s_r),
            word_to_token_indices(word_ids, o_l, o_r),
            word_to_token_indices(word_ids, p_l, p_r),
        )

        return {
            "tokens": tokens,
            "token_ids": encoding["input_ids"],
            "subject_span": token_triplets[0],
            "object_span": token_triplets[1],
            "predicate_span": token_triplets[2],
            "token_embeddings": token_embeddings,
        }


class SequenceLSOEIDataset(SpanLSOIEDataset):
    def __getitem__(self, idx: int) -> dict:
        """get tokens and triple for the given index

        Args:
            idx (int)

        Returns:
            dict: tokens, token_ids, spans
        """
        row = self.dataset.iloc[idx]
        words = row["words"]
        labels = row["label"]
        token_embs = row["token_embeddings"] if "token_embeddings" in row else None

        encoding = self.tokenizer(
            words, is_split_into_words=True, add_special_tokens=False
        )

        tokens = self.tokenizer.convert_ids_to_tokens(encoding["input_ids"])
        word_ids = encoding.word_ids()

        subject_labels = [i for i, t in enumerate(labels) if t.startswith("A0")]
        object_labels = [i for i, t in enumerate(labels) if t.startswith("A1")]
        predicate_labels = [i for i, t in enumerate(labels) if t.startswith("P")]

        subject_indices = [
            i for i, word_id in enumerate(word_ids) if word_id in subject_labels
        ]
        object_indices = [
            i for i, word_id in enumerate(word_ids) if word_id in object_labels
        ]
        predicate_indices = [
            i for i, word_id in enumerate(word_ids) if word_id in predicate_labels
        ]

        label = torch.zeros(len(tokens), dtype=torch.long)
        label[subject_indices] = 1
        label[object_indices] = 2
        label[predicate_indices] = 3

        # # sanity check
        # si = set(subject_indices)
        # oi = set(object_indices)
        # pi = set(predicate_indices)

        # if (si & oi) or (si & pi) or (oi & pi):
        #     raise ValueError("Subject, object, and predicate indices must be disjoint")
        # # sanity check end

        return {
            "tokens": tokens,
            "token_ids": encoding["input_ids"],
            "labels": label,
            "token_embeddings": token_embs,
        }


class SpanLSOEIDatasetConfig(BaseModel):
    """Configuration for SpanLSOIEDataset."""

    type: Literal["span"] = "span"
    tokenizer_name: str = "bert-base-uncased"
    filter_spans: bool = True
    use_cache: bool = False
    encoder: BERTEncoderConfig | None = None

    def create(self, split: str | list[str]) -> SpanLSOIEDataset:
        ds = SpanLSOIEDataset(
            split=split,
            tokenizer_name=self.tokenizer_name,
            filter_spans=self.filter_spans,
            encoder=self.encoder.create() if self.encoder is not None else None,
        )
        if self.use_cache:
            return CachedDataset(ds)
        else:
            return ds


class SequenceLSOEIDatasetConfig(BaseModel):
    """Configuration for SequenceLSOIEDataset."""

    type: Literal["sequence"] = "sequence"
    tokenizer_name: str = "bert-base-uncased"
    filter_spans: bool = False
    use_cache: bool = False
    encoder: BERTEncoderConfig | None = None

    def create(self, split: str | list[str]) -> SequenceLSOEIDataset:
        ds = SequenceLSOEIDataset(
            split=split,
            tokenizer_name=self.tokenizer_name,
            filter_spans=self.filter_spans,
            encoder=self.encoder.create() if self.encoder is not None else None,
        )
        if self.use_cache:
            return CachedDataset(ds)
        else:
            return ds


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s [%(name)s] %(message)s",
    )
    # ds = SequenceLSOEIDataset(split="train")
    ds = SpanLSOIEDataset(split="train")
    print(ds[0])

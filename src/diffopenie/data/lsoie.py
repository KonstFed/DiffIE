"""LSOIE dataset: base, span and sequence subclasses."""

import logging
import random
from typing import Literal

import pandas as pd
import torch
from datasets import load_dataset
from pydantic import BaseModel, ConfigDict
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from diffopenie.data import SEQ_STR2INT


logger = logging.getLogger(__name__)


def get_left_right_border(
    label: list[str], tag_prefix: str
) -> tuple[int | None, int | None]:
    """
    Return (left, right) span indices for the given tag prefix in a label sequence.

    Examples:
        example = ["O", "A0-B", "A0-I", "P-B", "P-I", "O", "A1-B", "A1-I"]
        get_left_right_border(example, "A0")  # (1, 2)
        get_left_right_border(example, "A1")  # (6, 7)
        get_left_right_border(example, "P")   # (3, 4)
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
) -> tuple[
    tuple[int | None, int | None],
    tuple[int | None, int | None],
    tuple[int | None, int | None],
]:
    """Word-level indices for A0 (subject), A1 (object), and P (predicate)."""
    return (
        get_left_right_border(label, "A0"),
        get_left_right_border(label, "A1"),
        get_left_right_border(label, "P"),
    )


def _is_label_continous(labels: list[str], prefix: str) -> bool:
    """Check if given label is continous"""
    first_index = next(
        (i for i, lab in enumerate(labels) if lab.startswith(prefix)), None
    )
    last_index = next(
        (i for i in range(len(labels) - 1, -1, -1) if labels[i].startswith(prefix)),
        None,
    )
    if first_index is None or last_index is None:
        return False
    return all(lab.startswith(prefix) for lab in labels[first_index : last_index + 1])


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


class LSOIEDataset(Dataset):
    """Base LSOIE dataset: loads splits only. No filtering, encoding, or __getitem__."""

    def __init__(
        self,
        split: str | list[str] = "train",
        drop_duplicate_sentences: bool = False, # TODO make to config
    ):
        splits = [split] if isinstance(split, str) else split
        self.split = splits

        hf_dataset = load_dataset("wardenga/lsoie", trust_remote_code=True)
        dfs = [pd.DataFrame(hf_dataset[s]) for s in splits]
        dataset = pd.concat(dfs, ignore_index=True).reset_index(drop=True)
        dataset["sentence"] = dataset["words"].apply(lambda x: " ".join(x))
        if drop_duplicate_sentences:
            n_before = len(dataset)
            dataset = dataset.drop_duplicates(subset=["sentence"], keep="first")
            n_dropped = n_before - len(dataset)
            if n_dropped:
                logger.info(
                    "Dropped %d duplicate sentence(s), keeping first occurrence.",
                    n_dropped,
                )
        self.dataset = dataset.sort_values(by="sentence").reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.dataset)

    def get_words_item(self, idx: int) -> dict[str, list]:
        """Return words, labels, and word-level spans for the given index."""
        row = self.dataset.iloc[idx]
        words = row["words"]
        labels = row["label"]

        (s_l, s_r), (o_l, o_r), (p_l, p_r) = labels_to_indices(labels)
        return {
            "words": words,
            "labels": labels,
            "subject_span": (s_l, s_r) if s_l is not None else None,
            "object_span": (o_l, o_r) if o_l is not None else None,
            "relation_span": (p_l, p_r) if p_l is not None else None,
        }


class SpanLSOIEDataset(LSOIEDataset):
    """LSOIE dataset with tokenization and token-level span encoding."""

    def __init__(
        self,
        split: str | list[str] = "train",
        tokenizer_name: str = "bert-base-uncased",
        filter_spans: bool = True,
        drop_duplicate_sentences: bool = False,
    ):
        super().__init__(split=split, drop_duplicate_sentences=drop_duplicate_sentences)
        if filter_spans:
            self._filter()
        else:
            logger.info("Span dataset filtering: disabled")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def _filter(self) -> None:
        """Filter triplets with None or not continous spans."""
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

    def __getitem__(self, idx: int) -> dict:
        """Return tokens and token-level spans for the given index."""
        row = self.get_words_item(idx)
        words = row["words"]
        s_l, s_r = row["subject_span"]
        o_l, o_r = row["object_span"]
        p_l, p_r = row["relation_span"]

        encoding = self.tokenizer(
            words, is_split_into_words=True, add_special_tokens=True
        )
        tokens = self.tokenizer.convert_ids_to_tokens(encoding["input_ids"])
        word_ids = encoding.word_ids()

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
        }


class SequenceLSOEIDataset(LSOIEDataset):
    """LSOIE dataset with sequence (token-level) labels instead of spans."""

    def __init__(
        self,
        split: str | list[str] = "train",
        tokenizer_name: str = "bert-base-uncased",
        filter_spans: bool = False,
        drop_duplicate_sentences: bool = False,
        max_rows: int | None = None,
        max_rows_seed: int = 0,
    ):
        super().__init__(split=split, drop_duplicate_sentences=drop_duplicate_sentences)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if max_rows is not None and max_rows < len(self.dataset):
            self.dataset = (
                self.dataset.sample(n=max_rows, random_state=max_rows_seed)
                .reset_index(drop=True)
            )
            logger.info(
                "SequenceLSOEIDataset: subsampled to %d rows (seed=%d)",
                max_rows, max_rows_seed,
            )

    def __getitem__(self, idx: int) -> dict:
        """Return tokens and token-level label tensor for the given index."""
        row = self.dataset.iloc[idx]
        words = row["words"]
        labels = row["label"]

        encoding = self.tokenizer(
            words, is_split_into_words=True, add_special_tokens=True
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
        label = label.fill_(SEQ_STR2INT["B"])
        label[subject_indices] = SEQ_STR2INT["S"]
        label[object_indices] = SEQ_STR2INT["O"]
        label[predicate_indices] = SEQ_STR2INT["R"]

        return {
            "tokens": tokens,
            "context_tokens": tokens,
            "context_token_ids": encoding["input_ids"],
            "tag_ids": label,
            "token_ids": encoding["input_ids"],
            "labels": label,
        }


class SpanLSOEIDatasetConfig(BaseModel):
    """Configuration for SpanLSOIEDataset."""

    model_config = ConfigDict(extra="forbid")
    type: Literal["span"] = "span"
    split: str | list[str] = "train"
    tokenizer_name: str = "bert-base-uncased"
    filter_spans: bool = True
    drop_duplicate_sentences: bool = False

    def create(self) -> SpanLSOIEDataset:
        return SpanLSOIEDataset(
            split=self.split,
            tokenizer_name=self.tokenizer_name,
            filter_spans=self.filter_spans,
            drop_duplicate_sentences=self.drop_duplicate_sentences,
        )


class SequenceLSOEIDatasetConfig(BaseModel):
    """Configuration for SequenceLSOEIDataset."""

    model_config = ConfigDict(extra="forbid")
    type: Literal["sequence"] = "sequence"
    split: str | list[str] = "train"
    tokenizer_name: str = "bert-base-uncased"
    filter_spans: bool = False
    drop_duplicate_sentences: bool = False
    max_rows: int | None = None
    max_rows_seed: int = 0

    def create(self) -> SequenceLSOEIDataset:
        return SequenceLSOEIDataset(
            split=self.split,
            tokenizer_name=self.tokenizer_name,
            filter_spans=self.filter_spans,
            drop_duplicate_sentences=self.drop_duplicate_sentences,
            max_rows=self.max_rows,
            max_rows_seed=self.max_rows_seed,
        )


class GroupedSequenceLSOEIDataset(SequenceLSOEIDataset):
    """Groups rows by sentence; each item returns `samples_per_group` random
    (target, prev-subset) pairs drawn from the group.

    Mirrors GroupedImojieDataset: the context is extended with the other
    triplets' "subject predicate object" text (each followed by [SEP]), while
    the tag sequence remains tied to the original sentence length.
    """

    def __init__(
        self,
        split: str | list[str] = "train",
        tokenizer_name: str = "bert-base-uncased",
        filter_spans: bool = False,
        drop_duplicate_sentences: bool = False,
        samples_per_group: int = 2,
    ):
        super().__init__(split, tokenizer_name, filter_spans, drop_duplicate_sentences)
        if samples_per_group < 1:
            raise ValueError("samples_per_group must be >= 1")
        self.samples_per_group = samples_per_group
        seen: dict[str, int] = {}
        self._groups: list[list[int]] = []
        for i, sentence in enumerate(self.dataset["sentence"].tolist()):
            if sentence not in seen:
                seen[sentence] = len(self._groups)
                self._groups.append([])
            self._groups[seen[sentence]].append(i)

    def __len__(self) -> int:
        return len(self._groups)

    def _triplet_text(self, df_idx: int) -> str:
        """Return 'subject predicate object' text for the row at df_idx."""
        row = self.dataset.iloc[df_idx]
        words = row["words"]
        labels = row["label"]
        (s_l, s_r), (o_l, o_r), (p_l, p_r) = labels_to_indices(labels)

        def _slice(left: int | None, right: int | None) -> str:
            if left is None or right is None:
                return ""
            return " ".join(words[left : right + 1])

        return f"{_slice(s_l, s_r)} {_slice(p_l, p_r)} {_slice(o_l, o_r)}".strip()

    def _encode_prev_triplets(self, df_indices: list[int]) -> tuple[list[int], list[str]]:
        """Tokenize prev-triplet text for the given df rows; returns (ids, tokens)."""
        sep_id = self.tokenizer.sep_token_id
        sep_tok = self.tokenizer.sep_token
        extra_ids, extra_toks = [], []
        for df_idx in df_indices:
            text = self._triplet_text(df_idx)
            if not text:
                continue
            enc = self.tokenizer(text, add_special_tokens=False)
            extra_ids += enc["input_ids"] + [sep_id]
            extra_toks += (
                self.tokenizer.convert_ids_to_tokens(enc["input_ids"]) + [sep_tok]
            )
        return extra_ids, extra_toks

    def _build_item(
        self, target_pos_in_group: int, group_df_indices: list[int]
    ) -> dict:
        target_idx = group_df_indices[target_pos_in_group]
        target = super().__getitem__(target_idx)
        pool = [
            group_df_indices[j]
            for j in range(len(group_df_indices))
            if j != target_pos_in_group
        ]
        selected = random.sample(pool, random.randint(0, len(pool)))
        extra_ids, extra_toks = self._encode_prev_triplets(selected)

        return {
            "tokens": list(target["tokens"]) + extra_toks,
            "context_tokens": list(target["context_tokens"]) + extra_toks,
            "context_token_ids": list(target["context_token_ids"]) + extra_ids,
            "tag_ids": target["tag_ids"],
            "token_ids": list(target["token_ids"]) + extra_ids,
            "labels": target["labels"],
        }

    def _build_stop_item(self, group_df_indices: list[int]) -> dict:
        """Stop sample: prev = all GT triplets (shuffled), target = all-B."""
        base = super().__getitem__(group_df_indices[0])
        shuffled = list(group_df_indices)
        random.shuffle(shuffled)
        extra_ids, extra_toks = self._encode_prev_triplets(shuffled)
        stop_tag = torch.zeros_like(base["tag_ids"])
        return {
            "tokens": list(base["tokens"]) + extra_toks,
            "context_tokens": list(base["context_tokens"]) + extra_toks,
            "context_token_ids": list(base["context_token_ids"]) + extra_ids,
            "tag_ids": stop_tag,
            "token_ids": list(base["token_ids"]) + extra_ids,
            "labels": stop_tag,
        }

    def __getitem__(self, idx: int) -> list[dict]:
        group_df_indices = self._groups[idx]
        n = len(group_df_indices)
        # n real targets + 1 stop slot → stop prob = 1/(n+1).
        out = []
        for _ in range(self.samples_per_group):
            choice = random.randrange(n + 1)
            if choice == n:
                out.append(self._build_stop_item(group_df_indices))
            else:
                out.append(self._build_item(choice, group_df_indices))
        return out


class GroupedSequenceLSOEIDatasetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["sequence_grouped"] = "sequence_grouped"
    split: str | list[str] = "train"
    tokenizer_name: str = "bert-base-uncased"
    filter_spans: bool = False
    drop_duplicate_sentences: bool = False
    samples_per_group: int = 2

    def create(self) -> GroupedSequenceLSOEIDataset:
        return GroupedSequenceLSOEIDataset(
            split=self.split,
            tokenizer_name=self.tokenizer_name,
            filter_spans=self.filter_spans,
            drop_duplicate_sentences=self.drop_duplicate_sentences,
            samples_per_group=self.samples_per_group,
        )

"""Imojie TSV with arg1/rel/arg2 tags; same output format as SequenceLSOEIDataset."""

import logging
import re
from pathlib import Path
from typing import Literal

import pandas as pd
import torch
from pydantic import BaseModel, ConfigDict
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from diffopenie.data import SEQ_STR2INT

logger = logging.getLogger(__name__)

def extract_triplet(label: str) -> tuple[str, str, str]:
    """Extract arg1, rel, arg2 from XML-style label string."""
    arg1_match = re.search(r"<arg1>(.*?)</arg1>", label)
    arg1 = arg1_match.group(1).strip() if arg1_match else ""

    rel_match = re.search(r"<rel>(.*?)</rel>", label)
    rel = rel_match.group(1).strip() if rel_match else ""

    arg2_match = re.search(r"<arg2>(.*?)</arg2>", label)
    arg2 = arg2_match.group(1).strip() if arg2_match else ""

    return arg1, rel, arg2


def label_to_sequence_labels(
    sentence: str, label: str
) -> tuple[list[str], list[str], float]:
    """
    Convert XML label to word-level labels: subject, relation, object, background.

    Returns (words, seq_labels, success_pct). success_pct is the fraction of
    label span words aligned to sentence words (0–100).
    """
    words = sentence.split()
    words_lower = [w.lower() for w in words]
    n = len(words)
    seq_labels = ["background"] * n
    sub, rel, obj = extract_triplet(label)

    def tokenize(s: str) -> list[str]:
        return s.split() if s else []

    sub_w = tokenize(sub)
    rel_w = tokenize(rel)
    obj_w = tokenize(obj)
    total_span_words = len(sub_w) + len(rel_w) + len(obj_w)
    matched_count = 0

    def find_contiguous(
        span_words: list[str], start_from: int
    ) -> tuple[int | None, int | None]:
        if not span_words:
            return None, None
        span_lower = [w.lower() for w in span_words]
        L = len(span_lower)
        for i in range(start_from, n - L + 1):
            if words_lower[i : i + L] == span_lower:
                return i, i + L
        return None, None

    def mark_range(start: int, end: int, tag: str) -> int:
        for j in range(start, end):
            seq_labels[j] = tag
        return end - start

    next_start = 0
    for span_words, tag in [
        (sub_w, "subject"),
        (rel_w, "relation"),
        (obj_w, "object"),
    ]:
        if not span_words:
            continue
        start_idx, end_idx = find_contiguous(span_words, next_start)
        if start_idx is not None:
            matched_count += mark_range(start_idx, end_idx, tag)
            next_start = end_idx
        else:
            for w in span_words:
                w_lower = w.lower()
                found = False
                for j in range(next_start, n):
                    if (
                        words_lower[j] == w_lower
                        and seq_labels[j] == "background"
                    ):
                        seq_labels[j] = tag
                        matched_count += 1
                        next_start = j + 1
                        found = True
                        break
                if not found:
                    next_start = n
                    break

    success_pct = (
        (100.0 * matched_count / total_span_words)
        if total_span_words
        else 100.0
    )
    return words, seq_labels, success_pct

# Map imojie tag names to same indices as LSOIE (B=0, S=1, R=2, O=3)
IMOJIE_TAG_TO_IDX = {
    "background": SEQ_STR2INT["B"],
    "subject": SEQ_STR2INT["S"],
    "relation": SEQ_STR2INT["R"],
    "object": SEQ_STR2INT["O"],
}


def _load_imojie_tsv(path: str | Path) -> pd.DataFrame:
    """Load single TSV file (no header; col 0=sentence, col 1=label)."""
    fp = Path(path)
    if not fp.exists():
        raise FileNotFoundError(f"Imojie TSV not found: {fp}")
    df = pd.read_csv(fp, sep="\t", header=None)
    df = df[[0, 1]].rename(columns={0: "sentence", 1: "label"})
    return df.reset_index(drop=True)


class SequenceImojieDataset(Dataset):
    """
    Sequence dataset from a single Imojie TSV file; output matches SequenceLSOEIDataset.

    path is the path to the TSV file (no header; column 0=sentence, 1=XML label).
    """

    def __init__(
        self,
        path: str | Path,
        tokenizer_name: str = "bert-base-uncased",
        min_success_pct: float | None = None,
    ):
        self.path = str(path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.min_success_pct = min_success_pct
        self._df = _load_imojie_tsv(path)
        if min_success_pct is not None:
            self._filter_by_success(min_success_pct)

    def _filter_by_success(self, min_pct: float) -> None:
        total = len(self._df)
        to_drop = []
        for i in range(total):
            row = self._df.iloc[i]
            _, _, pct = label_to_sequence_labels(
                row["sentence"], row["label"]
            )
            if pct < min_pct:
                to_drop.append(i)
        dropped = len(to_drop)
        purged_pct = (100.0 * dropped / total) if total else 0.0
        print(
            f"Imojie [{self.path}]: purged {dropped}/{total} rows "
            f"({purged_pct:.2f}%) with alignment < {min_pct:.1f}%"
        )
        if to_drop:
            self._df = self._df.drop(to_drop).reset_index(drop=True)

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, idx: int) -> dict:
        """Same keys as SequenceLSOEIDataset."""
        row = self._df.iloc[idx]
        sentence = row["sentence"]
        label_str = row["label"]
        words, seq_labels, _ = label_to_sequence_labels(sentence, label_str)
        word_level_labels = [
            IMOJIE_TAG_TO_IDX.get(s, SEQ_STR2INT["B"]) for s in seq_labels
        ]

        encoding = self.tokenizer(
            words, is_split_into_words=True, add_special_tokens=True
        )
        tokens = self.tokenizer.convert_ids_to_tokens(encoding["input_ids"])
        word_ids = encoding.word_ids()

        label = torch.zeros(len(tokens), dtype=torch.long)
        label = label.fill_(SEQ_STR2INT["B"])
        for i, wid in enumerate(word_ids):
            if wid is not None and wid < len(word_level_labels):
                label[i] = word_level_labels[wid]

        return {
            "tokens": tokens,
            "token_ids": encoding["input_ids"],
            "labels": label,
            "token_embeddings": None,
        }


class SequenceImojieDatasetConfig(BaseModel):
    """Configuration for SequenceImojieDataset."""

    model_config = ConfigDict(extra="forbid")
    type: Literal["imojie"] = "imojie"
    path: str
    tokenizer_name: str = "bert-base-uncased"
    min_success_pct: float | None = None

    def create(self) -> SequenceImojieDataset:
        return SequenceImojieDataset(
            path=self.path,
            tokenizer_name=self.tokenizer_name,
            min_success_pct=self.min_success_pct,
        )

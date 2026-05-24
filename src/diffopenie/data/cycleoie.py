"""CycleOIE TSV (source/target columns); output matches SequenceImojieDataset."""

import logging
import re
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict
from transformers import AutoTokenizer

from diffopenie.data.imojie import (
    SequenceImojieDataset,
    label_to_sequence_labels,
)

logger = logging.getLogger(__name__)


_TRIPLET_RE = re.compile(
    r"subject\s*<is>\s*(.*?)\s*<and>\s*"
    r"relation\s*<is>\s*(.*?)\s*<and>\s*"
    r"object\s*<is>\s*(.*?)\s*$"
)


def parse_cycleoie_target(target: str) -> list[tuple[str, str, str]]:
    """Split target on `<then>`; parse each chunk into (subject, relation, object).

    Handles both spaced (`subject <is> X`) and unspaced (`subject<is>X`) variants.
    """
    triplets: list[tuple[str, str, str]] = []
    for chunk in target.split("<then>"):
        chunk = chunk.strip()
        if not chunk:
            continue
        m = _TRIPLET_RE.match(chunk)
        if not m:
            continue
        sub, rel, obj = (g.strip() for g in m.groups())
        triplets.append((sub, rel, obj))
    return triplets


def _to_xml_label(sub: str, rel: str, obj: str) -> str:
    return f"<arg1>{sub}</arg1> <rel>{rel}</rel> <arg2>{obj}</arg2>"


def _load_cycleoie_tsv(
    path: str | Path, splits: list[str] | None = None
) -> pd.DataFrame:
    fp = Path(path)
    if not fp.exists():
        raise FileNotFoundError(f"CycleOIE file not found: {fp}")
    df = pd.read_csv(fp, sep="\t")
    if "source" not in df.columns or "target" not in df.columns:
        raise ValueError(
            f"CycleOIE expected 'source' and 'target' columns, got {list(df.columns)}"
        )
    if "split" in df.columns:
        # Safe default: keep only train rows so LSOIE dev/test don't leak into
        # training. Pass splits=["train", "dev", "test"] explicitly to opt out.
        keep = splits if splits is not None else ["train"]
        before = len(df)
        df = df[df["split"].isin(keep)].reset_index(drop=True)
        logger.info(
            "CycleOIE [%s]: filtered to splits=%s (%d/%d rows kept)",
            fp, keep, len(df), before,
        )
    elif splits is not None:
        logger.warning(
            "CycleOIE [%s]: 'splits' given but TSV has no 'split' column; "
            "filter ignored.", fp,
        )
    return df[["source", "target"]].reset_index(drop=True)


class CycleOIEDataset(SequenceImojieDataset):
    """Each `<then>`-separated triplet is fanned out to its own row, so the
    resulting items behave identically to SequenceImojieDataset (one
    (sentence, triplet) per __getitem__)."""

    def __init__(
        self,
        path: str | Path,
        tokenizer_name: str = "bert-base-uncased",
        min_success_pct: float | None = None,
        splits: list[str] | None = None,
        max_rows: int | None = None,
        max_rows_seed: int = 0,
    ):
        self.path = str(path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.min_success_pct = min_success_pct
        self.splits = splits

        raw = _load_cycleoie_tsv(path, splits=splits)
        if max_rows is not None and max_rows < len(raw):
            raw = raw.sample(
                n=max_rows, random_state=max_rows_seed
            ).reset_index(drop=True)
            logger.info(
                "CycleOIE [%s]: subsampled to %d source rows (seed=%d)",
                path, max_rows, max_rows_seed,
            )
        rows: list[dict] = []
        for _, r in raw.iterrows():
            sent = r["source"]
            for sub, rel, obj in parse_cycleoie_target(r["target"]):
                rows.append(
                    {"sentence": sent, "label": _to_xml_label(sub, rel, obj)}
                )
        self._df = pd.DataFrame(rows, columns=["sentence", "label"])

        if min_success_pct is not None:
            self._filter_by_success(min_success_pct)

    def _filter_by_success(self, min_pct: float) -> None:
        total = len(self._df)
        to_drop = []
        for i in range(total):
            row = self._df.iloc[i]
            _, _, pct = label_to_sequence_labels(row["sentence"], row["label"])
            if pct < min_pct:
                to_drop.append(i)
        dropped = len(to_drop)
        purged_pct = (100.0 * dropped / total) if total else 0.0
        print(
            f"CycleOIE [{self.path}]: purged {dropped}/{total} rows "
            f"({purged_pct:.2f}%) with alignment < {min_pct:.1f}%"
        )
        if to_drop:
            self._df = self._df.drop(to_drop).reset_index(drop=True)


class CycleOIEDatasetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["cycleoie"] = "cycleoie"
    path: str
    tokenizer_name: str = "bert-base-uncased"
    min_success_pct: float | None = None
    splits: list[str] | None = None
    max_rows: int | None = None
    max_rows_seed: int = 0

    def create(self) -> CycleOIEDataset:
        return CycleOIEDataset(
            path=self.path,
            tokenizer_name=self.tokenizer_name,
            min_success_pct=self.min_success_pct,
            splits=self.splits,
            max_rows=self.max_rows,
            max_rows_seed=self.max_rows_seed,
        )

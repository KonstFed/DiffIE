import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import BertTokenizerFast
import pandas as pd


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


class SpanLSOIEDataset(Dataset):
    """Dataset for the LSOIE dataset."""

    def __init__(self, split: str = "train", tokenizer_name: str = "bert-base-uncased"):
        self.split = split
        self.tokenizer = BertTokenizerFast.from_pretrained(
            tokenizer_name, use_fast=True
        )

        dataset = load_dataset("wardenga/lsoie", trust_remote_code=True)[split]
        dataset = pd.DataFrame(dataset)
        dataset["sentence"] = dataset["words"].apply(lambda x: " ".join(x))
        self.dataset = dataset.sort_values(by="sentence")

    def __len__(self) -> int:
        return len(self.dataset)

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

        # Get word-level indices for A0, A1, P
        (s_l, s_r), (o_l, o_r), (p_l, p_r) = labels_to_indices(labels)

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
            "spans": token_triplets,  # ((s_start, s_end), (o_start, o_end), (p_start, p_end))
            # "word_ids": word_ids,
            # "words": words,
            # "labels": labels,
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

        # sanity check
        si = set(subject_indices)
        oi = set(object_indices)
        pi = set(predicate_indices)

        if (si & oi) or (si & pi) or (oi & pi):
            raise ValueError("Subject, object, and predicate indices must be disjoint")
        # sanity check end

        return {
            "tokens": tokens,
            "token_ids": encoding["input_ids"],
            "labels": label,
        }


if __name__ == "__main__":
    ds = SequenceLSOEIDataset(split="train")
    print(ds[0])
    # for i in range(len(ds)):
    #     print(ds[i])

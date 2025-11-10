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


def labels_to_indices(label: list[str]) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    """Convert labels to word-level indices for A0 (subject), A1 (object), and P (predicate)."""
    return (
        get_left_right_border(label, "A0"),
        get_left_right_border(label, "A1"),
        get_left_right_border(label, "P")
    )


def word_to_token_indices(word_ids: list[int | None], word_start: int, word_end: int) -> tuple[int | None, int | None]:
    """Convert word indices to token indices."""
    if word_start is None or word_end is None:
        return None, None
    
    token_start = next((i for i, wid in enumerate(word_ids) if wid == word_start), None)
    token_end = next((i for i in range(len(word_ids)-1, -1, -1) if word_ids[i] == word_end), None)
    return token_start, token_end


class LSOIEDataset(Dataset):
    def __init__(self, split: str = "train", tokenizer_name: str = "bert-base-uncased"):
        self.split = split
        self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name)
        
        dataset = load_dataset("wardenga/lsoie")[split]
        dataset = pd.DataFrame(dataset)
        dataset["sentence"] = dataset["words"].apply(lambda x: " ".join(x))
        self.dataset = dataset.sort_values(by="sentence")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset.iloc[idx]
        words = row["words"]
        labels = row["label"]
        
        # Get word-level indices for A0, A1, P
        (s_l, s_r), (o_l, o_r), (p_l, p_r) = labels_to_indices(labels)
        
        # Tokenize words
        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            add_special_tokens=False
        )
        
        tokens = self.tokenizer.convert_ids_to_tokens(encoding["input_ids"])
        word_ids = encoding.word_ids()
        
        # Convert word indices to token indices
        token_triplets = (
            word_to_token_indices(word_ids, s_l, s_r),
            word_to_token_indices(word_ids, o_l, o_r),
            word_to_token_indices(word_ids, p_l, p_r)
        )
        
        return {
            "tokens": tokens,
            "token_ids": encoding["input_ids"],
            "spans": token_triplets,  # ((s_start, s_end), (o_start, o_end), (p_start, p_end))
            "word_ids": word_ids,
            "words": words,
            "labels": labels,
        }


import pytest
import torch

from diffopenie.data.triplet_utils import extract_longest_span


@pytest.mark.parametrize(
    "word_ids,pred_indices,expected,test_id",
    [
        # Basic case with a single continuous span
        ([0, 0, 1, 1, 2, 2], [True, True, True, True, False, False], (0, 1), "simple"),
        # Finding the longest run when multiple spans exist
        (
            [0, 0, 1, 1, 2, 2, 3, 3],
            [True, True, True, True, False, False, True, True],
            (0, 1),
            "longest_run",
        ),
        # Single word predicted
        (
            [0, 0, 1, 1, 2, 2],
            [False, False, True, True, False, False],
            (1, 1),
            "single_word",
        ),
        # No words are fully predicted
        (
            [0, 0, 1, 1, 2, 2],
            [False, False, False, False, False, False],
            None,
            "no_predictions",
        ),
        # Partially predicted words are not included
        ([0, 0, 1, 1, 1], [True, True, True, False, True], (0, 0), "partial_word"),
        # Empty word_ids
        ([], [], None, "empty"),
        # Words of different token lengths
        (
            [0, 1, 1, 2, 2, 2],
            [True, True, True, True, True, True],
            (0, 2),
            "uneven_lengths",
        ),
        # Gap in the middle of predictions
        (
            [0, 0, 1, 1, 2, 2, 3, 3],
            [True, True, False, False, False, False, True, True],
            (0, 0),
            "gap_in_middle",
        ),
    ],
)
def test_extract_longest_span(word_ids, pred_indices, expected, test_id):
    """Test extract_longest_span with various scenarios."""
    pred_tensor = torch.tensor(pred_indices)
    result = extract_longest_span(pred_tensor, word_ids)
    assert result == expected

if __name__ == "__main__":
    pred_indices = [True, True, True, True, True, True]
    word_ids = [0, 1, 1, 2, 2, 2]
    pred_tensor = torch.tensor(pred_indices)
    result = extract_longest_span(pred_tensor, word_ids)
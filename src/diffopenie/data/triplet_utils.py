import torch

def longest_true_run(x: torch.Tensor) -> tuple[int, int] | None:
    """
    Find the longest run of True values in a 1D boolean tensor.
    """
    x = x.to(torch.int32)

    padded = torch.cat([torch.tensor([0]), x, torch.tensor([0])])

    # find run boundaries
    diff = padded[1:] - padded[:-1]

    starts = torch.where(diff == 1)[0]
    ends = torch.where(diff == -1)[0] - 1

    if len(starts) == 0:
        return None  # no True values

    lengths = ends - starts + 1
    i = torch.argmax(lengths)

    return int(starts[i]), int(ends[i])


def extract_longest_span(pred_indices: torch.BoolTensor, word_ids: list[int | None]) -> tuple[int, int] | None:
    """
    Extract the longest span, which will not split any word.
    """
    word_starts = []
    prev = None

    for i, w in enumerate(word_ids):
        if w is not None and w != prev:
            word_starts.append(i)
        prev = w

    per_word_pred = []
    for i in range(len(word_starts)):
        start = word_starts[i]
        if i == len(word_starts) - 1:
            end = len(pred_indices)
        else:
            end = word_starts[i+1]
        per_word_pred.append(pred_indices[start:end].all())
    return longest_true_run(torch.tensor(per_word_pred))

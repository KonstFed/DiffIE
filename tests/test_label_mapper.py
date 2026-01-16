import torch

from diffopenie.models.label_mapper import LabelMapper


def test_label_mapper():
    """simple test."""
    num_classes = 7
    hidden_size = 768
    label_mapper = LabelMapper(num_classes, hidden_size)
    labels = torch.randint(0, num_classes, (10,))
    embeddings = label_mapper(labels)

    # multiplication should not change anything
    embeddings = embeddings * 1000

    assert (label_mapper.reverse(embeddings) == labels).all()

import torch
from torch import nn


class LabelMapper(nn.Module):
    """Map labels to embeddings and vice verca."""

    def __init__(self, num_classes: int, embedding_dim: int):
        super().__init__()
        self.num_classes = num_classes
        self.embs = nn.Embedding(num_classes, embedding_dim=embedding_dim)

    def forward(self, labels: torch.LongTensor) -> torch.FloatTensor:
        embeddings = self.embs(labels)
        return embeddings

    def reverse(self, embeddings: torch.FloatTensor) -> torch.LongTensor:
        """Reverse the mapping from embeddings to labels.

        We use cosine similarity to find the closest label to the embedding.

        Args:
            embeddings (torch.FloatTensor): Embeddings to map to labels [batch_size, embedding_dim]

        Returns:
            torch.LongTensor: Labels [batch_size]
        """
        # get embedding weights: (num_classes, embedding_dim)
        weight = self.embs.weight  # shape: (num_classes, embedding_dim)
        # Normalize embeddings and weights to unit vectors
        normed_inputs = torch.nn.functional.normalize(embeddings, p=2, dim=-1)  # (..., embedding_dim)
        normed_weights = torch.nn.functional.normalize(weight, p=2, dim=-1)     # (num_classes, embedding_dim)
        # Compute cosine similarity: (..., num_classes)
        # Cosine distance = 1 - cosine similarity, so argmax of cosine similarity gives min cosine distance
        sim = torch.matmul(normed_inputs, normed_weights.t())  # (..., num_classes)
        # For each input embedding, get label with highest cosine similarity (i.e., lowest cosine distance)
        labels = torch.argmax(sim, dim=-1)
        return labels


if __name__ == "__main__":
    # simple test

    num_classes = 7
    hidden_size = 768
    label_mapper = LabelMapper(num_classes, hidden_size)
    labels = torch.randint(0, num_classes, (10,))
    embeddings = label_mapper(labels)
    embeddings = embeddings * 1000
    print(embeddings.shape)
    print("labels:", labels)
    print("labels after reverse:", label_mapper.reverse(embeddings))

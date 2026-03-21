from abc import ABC, abstractmethod
from collections import defaultdict

Span = tuple[int, int] | None
Triplet = tuple[Span, Span, Span]


class BaseTripletModel(ABC):
    carb_k: int = 64
    carb_topk: int = 20

    @abstractmethod
    def get_triplets(self, words: list[list[str]]) -> list[Triplet]:
        """Get triplets from a batch of sentences given by words."""

    def full_triplet_distribution(
        self,
        words: list[str],
        k: int,
    ) -> tuple[list[Triplet], list[float]]:
        """Extract full triplet distribution from a sentence given by words.

        Args:
            words (list[str]): sentence
            k (int): number of samples to draw

        Returns:
            tuple[list[Triplet], list[float]]: list of triplets and
            list of probabilities in descending order of probability
        """
        candidates = self.get_triplets([words] * k)
        freq = defaultdict(int)

        for cand_triple in candidates:
            freq[cand_triple] += 1

        freq = {t: v / k for t, v in freq.items()}
        freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        triplets, probs = zip(*freq)
        return triplets, probs

    def get_carb_prediction(
        self,
        words: list[str],
    ) -> tuple[list[Triplet], list[float]]:
        """Get CARB predictions for a sentence using self.carb_k samples and self.carb_topk."""
        triplets, probs = self.full_triplet_distribution(words, k=self.carb_k)
        n = min(self.carb_topk, len(triplets))
        return list(triplets[:n]), list(probs[:n])

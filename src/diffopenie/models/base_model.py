from abc import ABC, abstractmethod
from collections import defaultdict

Triplet = tuple[tuple[int, int], tuple[int, int], tuple[int, int]]


class BaseTripletModel(ABC):
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

        # for item in freq.items():
        #     print(item)
        freq = {t: v / k for t, v in freq.items()}
        freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        triplets, probs = zip(*freq)
        return triplets, probs

    def get_carb_prediction(
        self,
        words: list[str],
        k: int,
        topk: int = 20,
    ) -> tuple[Triplet, float]:
        """Get CARB prediction for a sentence given by words."""
        triplets, probs = self.full_triplet_distribution(words, k)
        return triplets[: min(topk, len(triplets))], probs[: min(topk, len(probs))]

from abc import ABC, abstractmethod


class BaseTripletModel(ABC):
    @abstractmethod
    def get_triplets(
        self, words: list[list[str]]
    ) -> list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]]:
        """Get triplets from a batch of sentences given by words."""
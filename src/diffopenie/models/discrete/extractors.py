from collections import defaultdict
from typing import Annotated, Callable, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

Span = tuple[int, int] | None
Triplet = tuple[Span, Span, Span]

GetTripletsFn = Callable[[list[list[str]]], list[Triplet]]


class FrequencyExtractor:
    """
    Draws k diffusion samples and returns the most frequent triplets as a
    probability distribution (frequency / k), sorted by descending probability.
    """

    def __init__(self, k: int = 64, topk: int = 20):
        self.k = k
        self.topk = topk

    def get_carb_prediction(
        self,
        words: list[str],
        get_triplets_fn: GetTripletsFn,
    ) -> tuple[list[Triplet], list[float]]:
        candidates = get_triplets_fn([words] * self.k)
        freq: dict[Triplet, int] = defaultdict(int)
        for t in candidates:
            freq[t] += 1
        probs = {t: v / self.k for t, v in freq.items()}
        sorted_items = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        n = min(self.topk, len(sorted_items))
        if not sorted_items:
            return [], []
        triplets, confidences = zip(*sorted_items[:n])
        return list(triplets), list(confidences)


class FrequencyExtractorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["frequency"] = "frequency"
    k: int = 64
    topk: int = 20

    def create(self) -> FrequencyExtractor:
        return FrequencyExtractor(k=self.k, topk=self.topk)


ExtractorConfig = Annotated[
    Union[FrequencyExtractorConfig],
    Field(discriminator="type"),
]

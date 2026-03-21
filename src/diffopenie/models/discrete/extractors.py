from collections import defaultdict
from typing import Annotated, Literal, Protocol, Union

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth
from sklearn.decomposition import PCA

Span = tuple[int, int] | None
Triplet = tuple[Span, Span, Span]
SpanEmbs = tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]


class GetTripletsFn(Protocol):
    def __call__(
        self,
        words: list[list[str]],
        *,
        n: int = 1,
        return_span_embs: bool = False,
    ) -> list[Triplet] | tuple[list[Triplet], list[SpanEmbs]]: ...


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
        candidates = get_triplets_fn([words], n=self.k)
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


class _ClusterExtractorBase:
    use_span_embs: bool

    def _triplet_to_vec(self, triplet: Triplet) -> list[float]:
        sub, rel, obj = triplet
        return [sub[0], sub[1], rel[0], rel[1], obj[0], obj[1]]

    def _vec_to_triplet(self, vec: np.ndarray) -> Triplet:
        v = [max(0, round(float(x))) for x in vec]
        return (v[0], v[1]), (v[2], v[3]), (v[4], v[5])

    def _collect_valid(
        self,
        candidates: list[Triplet],
        embs: list[SpanEmbs] | None,
    ) -> tuple[list[Triplet], list[SpanEmbs] | None]:
        if embs is None:
            return [t for t in candidates if all(s is not None for s in t)], None
        pairs = [(t, e) for t, e in zip(candidates, embs) if all(s is not None for s in t)]
        if not pairs:
            return [], None
        valid_t, valid_e = zip(*pairs)
        return list(valid_t), list(valid_e)

    def _build_vecs(
        self,
        valid_triplets: list[Triplet],
        valid_embs: list[SpanEmbs] | None,
    ) -> np.ndarray:
        if not self.use_span_embs or valid_embs is None:
            return np.array([self._triplet_to_vec(t) for t in valid_triplets], dtype=float)

        D = next(
            (e.shape[0] for emb_tuple in valid_embs for e in emb_tuple if e is not None),
            None,
        )
        if D is None:
            return np.array([self._triplet_to_vec(t) for t in valid_triplets], dtype=float)

        rows = []
        for sub_e, obj_e, pred_e in valid_embs:
            present = [e.numpy() for e in (sub_e, obj_e, pred_e) if e is not None]
            rows.append(np.mean(present, axis=0) if present else np.zeros(D))
        return np.array(rows, dtype=float)

    def _representatives(
        self,
        labels: np.ndarray,
        valid_triplets: list[Triplet],
        centroids: np.ndarray,
        n_clusters: int,
    ) -> list[Triplet]:
        if not self.use_span_embs:
            # Centroids are in index space — decode directly.
            return [self._vec_to_triplet(centroids[i]) for i in range(n_clusters)]
        # Centroids are in embedding space and not decodable as spans.
        # Return the most frequent triplet among each cluster's members.
        reps = []
        for c in range(n_clusters):
            freq: dict[Triplet, int] = defaultdict(int)
            for t, lbl in zip(valid_triplets, labels):
                if lbl == c:
                    freq[t] += 1
            reps.append(max(freq, key=lambda t: freq[t]))
        return reps

    def _results_from_labels(
        self,
        labels: np.ndarray,
        valid_triplets: list[Triplet],
        centroids: np.ndarray,
        n_clusters: int,
    ) -> tuple[list[Triplet], list[float]]:
        counts = np.bincount(labels, minlength=n_clusters)
        confidences = (counts / len(valid_triplets)).tolist()
        reps = self._representatives(labels, valid_triplets, centroids, n_clusters)
        order = np.argsort(confidences)[::-1]
        return [reps[i] for i in order], [confidences[i] for i in order]


class KMeansExtractor(_ClusterExtractorBase):
    """
    Draws n_samples diffusion samples, clusters the resulting triplets with KMeans,
    and returns one representative triplet per cluster sorted by cluster size.

    When use_span_embs=False (default): clusters in 6-dim span-index space; centroid
    is rounded back to a triplet.
    When use_span_embs=True: clusters in BERT embedding space (concat or sum of span
    embeddings); representative is the most frequent triplet in each cluster.
    """

    def __init__(
        self,
        n_samples: int = 64,
        n_clusters: int = 10,
        use_span_embs: bool = False,
    ):
        self.n_samples = n_samples
        self.n_clusters = n_clusters
        self.use_span_embs = use_span_embs

    def get_carb_prediction(
        self,
        words: list[str],
        get_triplets_fn: GetTripletsFn,
    ) -> tuple[list[Triplet], list[float]]:
        raw = get_triplets_fn([words], n=self.n_samples, return_span_embs=self.use_span_embs)
        if self.use_span_embs:
            candidates, embs = raw  # type: ignore[misc]
        else:
            candidates, embs = raw, None  # type: ignore[assignment]

        valid, valid_embs = self._collect_valid(candidates, embs)
        if not valid:
            return [], []

        vecs = self._build_vecs(valid, valid_embs)
        n_clusters = min(self.n_clusters, len(vecs))
        kmeans = KMeans(n_clusters=n_clusters, n_init="auto", random_state=0)
        labels = kmeans.fit_predict(vecs)
        return self._results_from_labels(labels, valid, kmeans.cluster_centers_, n_clusters)


class MeanShiftExtractor(_ClusterExtractorBase):
    """
    Draws n_samples diffusion samples, clusters the resulting triplets with Mean Shift.
    The number of clusters is determined automatically from data density via bandwidth
    estimation — no n_clusters needed.

    When use_span_embs=False (default): clusters in 6-dim span-index space; centroid
    is rounded back to a triplet.
    When use_span_embs=True: clusters in BERT embedding space (concat or sum of span
    embeddings); representative is the most frequent triplet in each cluster.
    """

    def __init__(
        self,
        n_samples: int = 64,
        bandwidth: float | None = None,
        use_span_embs: bool = False,
        pca_components: int | None = None,
    ):
        self.n_samples = n_samples
        self.bandwidth = bandwidth
        self.use_span_embs = use_span_embs
        self.pca_components = pca_components

    def get_carb_prediction(
        self,
        words: list[str],
        get_triplets_fn: GetTripletsFn,
    ) -> tuple[list[Triplet], list[float]]:
        raw = get_triplets_fn([words], n=self.n_samples, return_span_embs=self.use_span_embs)
        if self.use_span_embs:
            candidates, embs = raw  # type: ignore[misc]
        else:
            candidates, embs = raw, None  # type: ignore[assignment]

        valid, valid_embs = self._collect_valid(candidates, embs)
        if not valid:
            return [], []

        vecs = self._build_vecs(valid, valid_embs)
        if self.use_span_embs and self.pca_components is not None and self.pca_components < vecs.shape[1]:
            vecs = PCA(n_components=self.pca_components).fit_transform(vecs)

        bandwidth = self.bandwidth or estimate_bandwidth(vecs, quantile=0.3)
        if bandwidth <= 0:
            bandwidth = 1.0

        ms = MeanShift(bandwidth=bandwidth, bin_seeding=not self.use_span_embs)
        labels = ms.fit_predict(vecs)
        n_clusters = len(ms.cluster_centers_)
        return self._results_from_labels(labels, valid, ms.cluster_centers_, n_clusters)


class MeanShiftExtractorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["meanshift"] = "meanshift"
    n_samples: int = 64
    bandwidth: float | None = None
    use_span_embs: bool = False
    pca_components: int | None = None

    def create(self) -> MeanShiftExtractor:
        return MeanShiftExtractor(
            n_samples=self.n_samples,
            bandwidth=self.bandwidth,
            use_span_embs=self.use_span_embs,
            pca_components=self.pca_components,
        )


class KMeansExtractorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["kmeans"] = "kmeans"
    n_samples: int = 64
    n_clusters: int = 10
    use_span_embs: bool = False

    def create(self) -> KMeansExtractor:
        return KMeansExtractor(
            n_samples=self.n_samples,
            n_clusters=self.n_clusters,
            use_span_embs=self.use_span_embs,
        )


class FrequencyExtractorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["frequency"] = "frequency"
    k: int = 64
    topk: int = 20

    def create(self) -> FrequencyExtractor:
        return FrequencyExtractor(k=self.k, topk=self.topk)


ExtractorConfig = Annotated[
    Union[FrequencyExtractorConfig, KMeansExtractorConfig, MeanShiftExtractorConfig],
    Field(discriminator="type"),
]

from collections import defaultdict
from typing import Annotated, Literal, Protocol, Union

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth
from sklearn.decomposition import PCA

try:
    from sklearn.cluster import HDBSCAN
except ImportError:  # pragma: no cover
    from hdbscan import HDBSCAN  # type: ignore[no-redef]

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


def _bow_match(a: list[str], b: list[str]) -> int:
    """Greedy bag-of-words match count (multiset intersection size)."""
    b_rem = list(b)
    m = 0
    for w in a:
        if w in b_rem:
            m += 1
            b_rem.remove(w)
    return m


def _triplet_lenient_f1(
    a_fields: tuple[list[str], list[str], list[str]],
    b_fields: tuple[list[str], list[str], list[str]],
) -> float:
    """Symmetric F1 of CaRB-style lenient overlap over (sub, rel, obj)."""
    total_m = sum(_bow_match(x, y) for x, y in zip(a_fields, b_fields))
    total_a = sum(len(x) for x in a_fields)
    total_b = sum(len(y) for y in b_fields)
    if total_a == 0 or total_b == 0:
        return 0.0
    prec = total_m / total_b
    rec = total_m / total_a
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


class _UnionFind:
    __slots__ = ("parent",)

    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


class LenientFrequencyExtractor:
    """
    Like FrequencyExtractor, but merges near-duplicate surface variants before
    ranking. Algorithm:
      1. Draw k samples.
      2. Collapse identical-span triplets (exact dedup) → unique candidates
         with counts.
      3. Build the words-level text per field; cluster uniques via union-find
         where edges are pairs with symmetric lenient-F1 overlap ≥ threshold.
      4. For each cluster, sum mass = (Σ counts) / k; representative triplet
         = the highest-count member.
      5. Return top-k clusters sorted by mass.

    Catches cases like "Barack Obama; was born; Hawaii" vs "Obama; born;
    Hawaii" that exact-frequency splits across two buckets.
    """

    def __init__(self, k: int = 64, topk: int = 20, threshold: float = 0.7):
        self.k = k
        self.topk = topk
        self.threshold = threshold

    def get_carb_prediction(
        self,
        words: list[str],
        get_triplets_fn: GetTripletsFn,
    ) -> tuple[list[Triplet], list[float]]:
        candidates = get_triplets_fn([words], n=self.k)

        # 1. exact-string dedup with counts
        exact: dict[Triplet, int] = defaultdict(int)
        for t in candidates:
            exact[t] += 1
        uniques: list[Triplet] = list(exact.keys())
        counts: list[int] = [exact[t] for t in uniques]

        # 2. precompute lowercased word lists per field (None for invalid)
        def fields(
            t: Triplet,
        ) -> tuple[list[str], list[str], list[str]] | None:
            sub, obj, pred = t
            if sub is None or obj is None or pred is None:
                return None
            return (
                [w.lower() for w in words[sub[0] : sub[1] + 1]],
                [w.lower() for w in words[pred[0] : pred[1] + 1]],
                [w.lower() for w in words[obj[0] : obj[1] + 1]],
            )

        text_fields = [fields(t) for t in uniques]

        # 3. union-find on the unique set (invalid stays singleton)
        n_u = len(uniques)
        uf = _UnionFind(n_u)
        for i in range(n_u):
            if text_fields[i] is None:
                continue
            for j in range(i + 1, n_u):
                if text_fields[j] is None:
                    continue
                if (
                    _triplet_lenient_f1(text_fields[i], text_fields[j])
                    >= self.threshold
                ):
                    uf.union(i, j)

        # 4. aggregate mass per cluster
        groups: dict[int, list[int]] = defaultdict(list)
        for i in range(n_u):
            groups[uf.find(i)].append(i)

        results: list[tuple[Triplet, float]] = []
        for indices in groups.values():
            total = sum(counts[i] for i in indices)
            rep_idx = max(indices, key=lambda i: counts[i])
            results.append((uniques[rep_idx], total / self.k))

        # 5. sort + topk
        results.sort(key=lambda x: x[1], reverse=True)
        n_out = min(self.topk, len(results))
        if n_out == 0:
            return [], []
        triplets, probs = zip(*results[:n_out])
        return list(triplets), list(probs)


class LenientKDEExtractor:
    """
    Density-peak extractor over lenient triplet similarity.

    The extractor samples k triplets, collapses exact duplicates, computes a
    hard-KDE density around each unique triplet, picks local density peaks with
    non-maximum suppression, and assigns nearby triplets to the nearest peak.
    The number of returned extractions is dynamic and controlled indirectly by
    threshold.
    """

    def __init__(
        self,
        k: int = 64,
        threshold: float = 0.7,
    ):
        self.k = k
        self.threshold = threshold
        self.min_support = 2
        self.fallback_top1 = True

    def _fields(
        self, words: list[str], triplet: Triplet
    ) -> tuple[list[str], list[str], list[str]] | None:
        sub, obj, pred = triplet
        if sub is None or obj is None or pred is None:
            return None
        return (
            [w.lower() for w in words[sub[0] : sub[1] + 1]],
            [w.lower() for w in words[pred[0] : pred[1] + 1]],
            [w.lower() for w in words[obj[0] : obj[1] + 1]],
        )

    def get_carb_prediction(
        self,
        words: list[str],
        get_triplets_fn: GetTripletsFn,
    ) -> tuple[list[Triplet], list[float]]:
        candidates = get_triplets_fn([words], n=self.k)

        exact: dict[Triplet, int] = defaultdict(int)
        for t in candidates:
            if all(s is not None for s in t):
                exact[t] += 1

        if not exact:
            return [], []

        valid_pairs: list[
            tuple[Triplet, float, tuple[list[str], list[str], list[str]]]
        ] = []
        for t, c in exact.items():
            fields = self._fields(words, t)
            if fields is not None:
                valid_pairs.append((t, float(c), fields))
        if not valid_pairs:
            return [], []

        uniques: list[Triplet] = [t for t, _, _ in valid_pairs]
        counts = np.array([c for _, c, _ in valid_pairs], dtype=float)
        text_fields = [fields for _, _, fields in valid_pairs]
        valid_total = float(counts.sum())

        n_u = len(uniques)
        sim = np.eye(n_u, dtype=float)
        for i in range(n_u):
            for j in range(i + 1, n_u):
                s = _triplet_lenient_f1(text_fields[i], text_fields[j])
                sim[i, j] = s
                sim[j, i] = s

        rho = np.zeros(n_u, dtype=float)
        for i in range(n_u):
            rho[i] = counts[sim[i] >= self.threshold].sum()

        order = sorted(range(n_u), key=lambda i: (-rho[i], -counts[i], i))
        centers: list[int] = []
        for i in order:
            if rho[i] < self.min_support:
                break
            if any(sim[i, c] >= self.threshold for c in centers):
                continue
            centers.append(i)

        if not centers:
            top_idx = max(range(n_u), key=lambda i: (counts[i], rho[i], -i))
            conf = float(counts[top_idx] / valid_total) if valid_total > 0 else 0.0
            return [uniques[top_idx]], [conf]

        assignments: dict[int, list[int]] = {c: [] for c in centers}
        for i in range(n_u):
            best_center = max(centers, key=lambda c: (sim[i, c], -c))
            if sim[i, best_center] >= self.threshold:
                assignments.setdefault(best_center, []).append(i)

        results: list[tuple[Triplet, float]] = []
        for c in centers:
            members = assignments.get(c, [c])
            cluster_count = float(counts[members].sum())
            rep_idx = max(members, key=lambda i: (counts[i], rho[i], -i))
            confidence = cluster_count / valid_total if valid_total > 0 else 0.0
            results.append((uniques[rep_idx], confidence))

        results.sort(key=lambda x: x[1], reverse=True)

        triplets = [trip for trip, _ in results]
        confidences = [conf for _, conf in results]
        return triplets, confidences


class HDBSCANExtractor:
    """
    HDBSCAN clustering over lenient triplet similarity.

    Uses a precomputed distance matrix derived from the symmetric lenient F1
    similarity over triplet surface text. The number of returned extractions is
    dynamic and controlled by the HDBSCAN density parameters.
    """

    def __init__(
        self,
        k: int = 64,
        min_cluster_size: int = 2,
        min_samples: int | None = None,
        fallback_top1: bool = True,
    ):
        self.k = k
        self.min_cluster_size = max(2, min_cluster_size)
        self.min_samples = min_samples
        self.fallback_top1 = fallback_top1

    def _fields(
        self, words: list[str], triplet: Triplet
    ) -> tuple[list[str], list[str], list[str]] | None:
        sub, obj, pred = triplet
        if sub is None or obj is None or pred is None:
            return None
        return (
            [w.lower() for w in words[sub[0] : sub[1] + 1]],
            [w.lower() for w in words[pred[0] : pred[1] + 1]],
            [w.lower() for w in words[obj[0] : obj[1] + 1]],
        )

    def get_carb_prediction(
        self,
        words: list[str],
        get_triplets_fn: GetTripletsFn,
    ) -> tuple[list[Triplet], list[float]]:
        candidates = get_triplets_fn([words], n=self.k)

        exact: dict[Triplet, int] = defaultdict(int)
        for t in candidates:
            if all(s is not None for s in t):
                exact[t] += 1

        if not exact:
            return [], []

        valid_pairs: list[
            tuple[Triplet, float, tuple[list[str], list[str], list[str]]]
        ] = []
        for t, c in exact.items():
            fields = self._fields(words, t)
            if fields is not None:
                valid_pairs.append((t, float(c), fields))
        if not valid_pairs:
            return [], []

        uniques: list[Triplet] = [t for t, _, _ in valid_pairs]
        counts = np.array([c for _, c, _ in valid_pairs], dtype=float)
        text_fields = [fields for _, _, fields in valid_pairs]
        valid_total = float(counts.sum())

        n_u = len(uniques)
        sim = np.eye(n_u, dtype=float)
        for i in range(n_u):
            for j in range(i + 1, n_u):
                s = _triplet_lenient_f1(text_fields[i], text_fields[j])
                sim[i, j] = s
                sim[j, i] = s

        dist = 1.0 - sim
        if n_u < self.min_cluster_size:
            if not self.fallback_top1:
                return [], []
            top_idx = max(range(n_u), key=lambda i: (counts[i], -i))
            conf = float(counts[top_idx] / valid_total) if valid_total > 0 else 0.0
            return [uniques[top_idx]], [conf]

        clusterer = HDBSCAN(
            metric="precomputed",
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            cluster_selection_method="eom",
            allow_single_cluster=False,
        )
        labels = clusterer.fit_predict(dist)

        cluster_ids = sorted(lbl for lbl in set(labels) if lbl != -1)
        if not cluster_ids:
            if not self.fallback_top1:
                return [], []
            top_idx = max(range(n_u), key=lambda i: (counts[i], -i))
            conf = float(counts[top_idx] / valid_total) if valid_total > 0 else 0.0
            return [uniques[top_idx]], [conf]

        results: list[tuple[Triplet, float]] = []
        for lbl in cluster_ids:
            members = [i for i, y in enumerate(labels) if y == lbl]
            if not members:
                continue
            cluster_count = float(counts[members].sum())
            rep_idx = max(members, key=lambda i: (counts[i], -i))
            confidence = cluster_count / valid_total if valid_total > 0 else 0.0
            results.append((uniques[rep_idx], confidence))

        results.sort(key=lambda x: x[1], reverse=True)
        triplets = [trip for trip, _ in results]
        confidences = [conf for _, conf in results]
        return triplets, confidences


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
        pairs = [
            (t, e) for t, e in zip(candidates, embs) if all(s is not None for s in t)
        ]
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
            return np.array(
                [self._triplet_to_vec(t) for t in valid_triplets], dtype=float
            )

        D = next(
            (
                e.shape[0]
                for emb_tuple in valid_embs
                for e in emb_tuple
                if e is not None
            ),
            None,
        )
        if D is None:
            return np.array(
                [self._triplet_to_vec(t) for t in valid_triplets], dtype=float
            )

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
        raw = get_triplets_fn(
            [words], n=self.n_samples, return_span_embs=self.use_span_embs
        )
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
        return self._results_from_labels(
            labels, valid, kmeans.cluster_centers_, n_clusters
        )


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
        raw = get_triplets_fn(
            [words], n=self.n_samples, return_span_embs=self.use_span_embs
        )
        if self.use_span_embs:
            candidates, embs = raw  # type: ignore[misc]
        else:
            candidates, embs = raw, None  # type: ignore[assignment]

        valid, valid_embs = self._collect_valid(candidates, embs)
        if not valid:
            return [], []

        vecs = self._build_vecs(valid, valid_embs)
        if (
            self.use_span_embs
            and self.pca_components is not None
            and self.pca_components < vecs.shape[1]
        ):
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


class LenientFrequencyExtractorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["lenient_frequency"] = "lenient_frequency"
    k: int = 64
    topk: int = 20
    threshold: float = 0.7

    def create(self) -> LenientFrequencyExtractor:
        return LenientFrequencyExtractor(
            k=self.k, topk=self.topk, threshold=self.threshold
        )


class LenientKDEExtractorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["lenient_kde"] = "lenient_kde"
    k: int = 64
    threshold: float = 0.7

    def create(self) -> LenientKDEExtractor:
        return LenientKDEExtractor(k=self.k, threshold=self.threshold)


class HDBSCANExtractorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["hdbscan"] = "hdbscan"
    k: int = 64
    min_cluster_size: int = 2
    min_samples: int | None = None
    fallback_top1: bool = True

    def create(self) -> HDBSCANExtractor:
        return HDBSCANExtractor(
            k=self.k,
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            fallback_top1=self.fallback_top1,
        )


ExtractorConfig = Annotated[
    Union[
        FrequencyExtractorConfig,
        LenientFrequencyExtractorConfig,
        LenientKDEExtractorConfig,
        HDBSCANExtractorConfig,
        KMeansExtractorConfig,
        MeanShiftExtractorConfig,
    ],
    Field(discriminator="type"),
]

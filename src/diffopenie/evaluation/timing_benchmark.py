"""Measure DiffIE inference time split into model and clustering components."""

import argparse
import csv
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from diffopenie.evaluation.carb_eval import load_model
from diffopenie.models.discrete.extractors import (
    LenientFrequencyExtractor,
    Triplet,
    _triplet_lenient_f1,
)
from diffopenie.training.train_example import TrainingConfig
from diffopenie.utils import load_config, seed_everything


def _sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def _cluster_lenient_frequency(
    words: list[str],
    candidates: list[Triplet],
    extractor: LenientFrequencyExtractor,
) -> tuple[list[Triplet], list[float]]:
    """Run the same clustering/ranking logic as LenientFrequencyExtractor."""
    exact: dict[Triplet, int] = defaultdict(int)
    for triplet in candidates:
        exact[triplet] += 1
    uniques = list(exact.keys())
    counts = [exact[triplet] for triplet in uniques]

    def fields(triplet: Triplet) -> tuple[list[str], list[str], list[str]] | None:
        sub, obj, pred = triplet
        if sub is None or obj is None or pred is None:
            return None
        return (
            [word.lower() for word in words[sub[0] : sub[1] + 1]],
            [word.lower() for word in words[pred[0] : pred[1] + 1]],
            [word.lower() for word in words[obj[0] : obj[1] + 1]],
        )

    text_fields = [fields(triplet) for triplet in uniques]

    parent = list(range(len(uniques)))

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_left] = root_right

    for i in range(len(uniques)):
        if text_fields[i] is None:
            continue
        for j in range(i + 1, len(uniques)):
            if text_fields[j] is None:
                continue
            if (
                _triplet_lenient_f1(text_fields[i], text_fields[j])
                >= extractor.threshold
            ):
                union(i, j)

    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(len(uniques)):
        groups[find(i)].append(i)

    results: list[tuple[Triplet, float]] = []
    for indices in groups.values():
        total = sum(counts[i] for i in indices)
        rep_idx = max(indices, key=lambda i: counts[i])
        results.append((uniques[rep_idx], total / extractor.k))

    results.sort(key=lambda item: item[1], reverse=True)
    n_out = min(extractor.topk, len(results))
    if n_out == 0:
        return [], []
    triplets, probs = zip(*results[:n_out])
    return list(triplets), list(probs)


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def _summarize(
    *,
    sentences: list[str],
    k: int,
    repeat: int,
    model_times: list[float],
    clustering_times: list[float],
    device: torch.device,
) -> dict[str, Any]:
    total_times = [
        model_time + clustering_time
        for model_time, clustering_time in zip(model_times, clustering_times)
    ]
    total_seconds = sum(total_times)
    model_seconds = sum(model_times)
    clustering_seconds = sum(clustering_times)
    total_sentences = len(sentences)
    peak_mem_mb = ""
    if device.type == "cuda":
        peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

    return {
        "repeat": repeat,
        "num_sentences": total_sentences,
        "k": k,
        "total_model_seconds": model_seconds,
        "total_clustering_seconds": clustering_seconds,
        "total_seconds": total_seconds,
        "model_sec_per_sentence": model_seconds / total_sentences,
        "clustering_sec_per_sentence": clustering_seconds / total_sentences,
        "total_sec_per_sentence": total_seconds / total_sentences,
        "median_model_sec_per_sentence": _median(model_times),
        "median_clustering_sec_per_sentence": _median(clustering_times),
        "median_total_sec_per_sentence": _median(total_times),
        "model_percent": 100.0 * model_seconds / total_seconds
        if total_seconds
        else 0.0,
        "clustering_percent": 100.0 * clustering_seconds / total_seconds
        if total_seconds
        else 0.0,
        "sentences_per_second": total_sentences / total_seconds
        if total_seconds
        else 0.0,
        "device": str(device),
        "backend": device.type,
        "peak_cuda_mem_mb": peak_mem_mb,
    }


@torch.no_grad()
def run_timing(
    *,
    model,
    sentences: list[str],
    warmup_sentences: list[str],
    repeats: int,
    validate: int,
) -> list[dict[str, Any]]:
    extractor = model.extractor
    if not isinstance(extractor, LenientFrequencyExtractor):
        raise TypeError(
            "timing_benchmark currently supports only lenient_frequency "
            f"extractors, got {type(extractor).__name__}"
        )

    model.eval()
    device = model.device

    for sentence in tqdm(warmup_sentences, desc="Warmup", leave=False):
        words = sentence.split()
        candidates = model.get_triplets([words], n=extractor.k)
        _cluster_lenient_frequency(words, candidates, extractor)

    rows: list[dict[str, Any]] = []
    for repeat in range(1, repeats + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        model_times: list[float] = []
        clustering_times: list[float] = []
        checked = 0

        for sentence in tqdm(sentences, desc=f"Repeat {repeat}/{repeats}"):
            words = sentence.split()

            _sync_if_needed(device)
            model_start = time.perf_counter()
            candidates = model.get_triplets([words], n=extractor.k)
            _sync_if_needed(device)
            model_times.append(time.perf_counter() - model_start)

            cluster_start = time.perf_counter()
            triplets, probs = _cluster_lenient_frequency(words, candidates, extractor)
            clustering_times.append(time.perf_counter() - cluster_start)

            if checked < validate:
                expected_triplets, expected_probs = extractor.get_carb_prediction(
                    words,
                    lambda _words, *, n=1, return_span_embs=False: candidates,
                )
                if triplets != expected_triplets or probs != expected_probs:
                    raise AssertionError(
                        "split clustering output does not match extractor output"
                    )
                checked += 1

        rows.append(
            _summarize(
                sentences=sentences,
                k=extractor.k,
                repeat=repeat,
                model_times=model_times,
                clustering_times=clustering_times,
                device=device,
            )
        )

    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _print_summary(rows: list[dict[str, Any]]) -> None:
    model_sps = _mean([row["model_sec_per_sentence"] for row in rows])
    clustering_sps = _mean([row["clustering_sec_per_sentence"] for row in rows])
    total_sps = _mean([row["total_sec_per_sentence"] for row in rows])
    sent_per_sec = _mean([row["sentences_per_second"] for row in rows])
    model_pct = _mean([row["model_percent"] for row in rows])
    clustering_pct = _mean([row["clustering_percent"] for row in rows])

    print("\n--- timing summary (mean over repeats) ---")
    print(f"k:                         {rows[0]['k']}")
    print(f"sentences:                 {rows[0]['num_sentences']}")
    print(f"model sec/sentence:        {model_sps:.6f} ({model_pct:.1f}%)")
    print(
        f"clustering sec/sentence:   {clustering_sps:.6f} "
        f"({clustering_pct:.1f}%)"
    )
    print(f"total sec/sentence:        {total_sps:.6f}")
    print(f"end-to-end sentences/sec:  {sent_per_sec:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Time DiffIE model inference separately from lenient clustering."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--input-sentences", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--validate",
        type=int,
        default=3,
        help=(
            "Number of measured sentences per repeat to compare against "
            "extractor output."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override, e.g. cuda, cpu, or mps.",
    )
    args = parser.parse_args()

    seed_everything(args.seed)
    config = load_config(TrainingConfig, args.config)
    if args.device is not None:
        config.trainer.device = args.device

    print("Loading model...")
    model = load_model(config, args.checkpoint_path)
    if args.device is not None:
        model = model.to(args.device)
        if hasattr(model.scheduler, "to"):
            model.scheduler.to(args.device)
    model.eval()

    with args.input_sentences.open("r", encoding="utf-8") as f:
        all_sentences = [line.strip() for line in f if line.strip()]

    if args.limit is not None:
        all_sentences = all_sentences[: args.limit]
    if not all_sentences:
        raise ValueError("no input sentences to time")

    warmup_sentences = all_sentences[: min(args.warmup, len(all_sentences))]
    print(
        f"Timing {len(all_sentences)} sentence(s), "
        f"warmup={len(warmup_sentences)}, repeats={args.repeats}"
    )
    rows = run_timing(
        model=model,
        sentences=all_sentences,
        warmup_sentences=warmup_sentences,
        repeats=args.repeats,
        validate=args.validate,
    )
    _write_csv(args.out, rows)
    _print_summary(rows)
    print(f"\nWrote timing CSV to {args.out}")


if __name__ == "__main__":
    main()

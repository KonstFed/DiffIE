"""
Tests comparing our CaRB metrics reimplementation against the original CaRB repo.

Uses both realistic extractions and random/noisy ones to verify agreement.
"""

from __future__ import annotations

import os
import sys
import tempfile
from collections import defaultdict
from copy import copy
from pathlib import Path

import numpy as np
import pytest

from diffopenie.evaluation.carb_metrics import (
    CarbResult,
    Extraction,
    _binarize_extraction,
    _lenient_tuple_match,
    binary_lenient_match,
    evaluate,
)

# ── Path to the original CaRB repo ─────────────────────────────────────────

CARB_REPO = Path(__file__).resolve().parent.parent.parent / "CaRB"
CARB_GOLD = CARB_REPO / "data" / "gold" / "test.tsv"
HAS_CARB = CARB_REPO.exists() and CARB_GOLD.exists()


def _import_original_carb():
    """Import original CaRB modules by temporarily adding to sys.path."""
    carb_path = str(CARB_REPO)
    if carb_path not in sys.path:
        sys.path.insert(0, carb_path)

    # CaRB's extraction.py uses a removed sklearn import that's never actually
    # called — patch it out so import succeeds with modern sklearn.
    import types
    import importlib

    fake_mod = types.ModuleType("sklearn.preprocessing.data")
    fake_mod.binarize = None
    sys.modules["sklearn.preprocessing.data"] = fake_mod

    # Force re-import in case prior attempts cached the failure
    for mod_name in list(sys.modules):
        if mod_name.startswith("oie_readers") or mod_name in ("matcher", "carb"):
            del sys.modules[mod_name]

    from matcher import Matcher
    from oie_readers.extraction import Extraction as CarbExtraction
    from oie_readers.goldReader import GoldReader
    from oie_readers.tabReader import TabReader

    carb_mod = __import__("carb")
    return carb_mod.Benchmark, Matcher, CarbExtraction, GoldReader, TabReader


# ── Helpers ─────────────────────────────────────────────────────────────────


def _make_gold_tsv(extractions: dict[str, list[tuple[str, list[str]]]]) -> str:
    """Build gold TSV content: sent\\tpred\\targ1\\targ2..."""
    lines = []
    for sent, exs in extractions.items():
        for pred, args in exs:
            lines.append("\t".join([sent, pred] + args))
    return "\n".join(lines) + "\n"


def _make_pred_tsv(extractions: dict[str, list[tuple[float, str, list[str]]]]) -> str:
    """Build predicted TSV content: sent\\tconf\\tpred\\targ1\\targ2..."""
    lines = []
    for sent, exs in extractions.items():
        for conf, pred, args in exs:
            lines.append("\t".join([sent, str(conf), pred] + args))
    return "\n".join(lines) + "\n"


def _run_original_carb(gold_path: str, pred_path: str) -> tuple[float, tuple]:
    """Run the original CaRB benchmark and return (auc, (p, r, f1))."""
    Benchmark, Matcher, _, _, TabReader = _import_original_carb()
    b = Benchmark(gold_path)
    tr = TabReader()
    tr.read(pred_path)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".dat", delete=False) as f:
        out_fn = f.name
    try:
        auc_score, optimal = b.compare(
            predicted=tr.oie,
            matchingFunc=Matcher.binary_linient_tuple_match,
            output_fn=out_fn,
        )
    finally:
        os.unlink(out_fn)
    return float(auc_score), tuple(float(x) for x in optimal)


def _run_our_carb(gold_path: str, pred_path: str) -> CarbResult:
    """Run our CaRB reimplementation."""
    from diffopenie.evaluation.carb_metrics import (
        load_gold_file,
        load_predicted_file,
    )

    gold = load_gold_file(gold_path)
    pred = load_predicted_file(pred_path)
    return evaluate(gold, pred)


# ── Unit tests for matching functions ───────────────────────────────────────


class TestLenientTupleMatch:
    def test_exact_match(self):
        g = Extraction(pred="is", args=["John", "a teacher"])
        p = Extraction(pred="is", args=["John", "a teacher"])
        prec, rec = _lenient_tuple_match(g, p)
        assert prec == 1.0
        assert rec == 1.0

    def test_partial_match(self):
        g = Extraction(pred="is", args=["John Smith", "a good teacher"])
        p = Extraction(pred="is", args=["John", "a teacher"])
        prec, rec = _lenient_tuple_match(g, p)
        assert prec == 1.0  # all predicted words found in gold
        assert rec < 1.0  # not all gold words found

    def test_no_pred_match(self):
        g = Extraction(pred="runs", args=["John", "fast"])
        p = Extraction(pred="eats", args=["John", "fast"])
        prec, rec = _lenient_tuple_match(g, p)
        assert prec == 0.0
        assert rec == 0.0

    def test_be_form_matching(self):
        g = Extraction(pred="was", args=["He", "happy"])
        p = Extraction(pred="be", args=["He", "happy"])
        prec, rec = _lenient_tuple_match(g, p)
        assert prec > 0.0
        assert rec > 0.0

    def test_missing_arg_returns_zero(self):
        g = Extraction(pred="gave", args=["John", "Mary"])
        p = Extraction(pred="gave", args=["John"])
        prec, rec = _lenient_tuple_match(g, p)
        assert prec == 0.0
        assert rec == 0.0


class TestBinaryLenientMatch:
    def test_binarize_nary(self):
        ex = Extraction(pred="gave", args=["John", "a book", "to Mary"])
        b = _binarize_extraction(ex)
        assert len(b.args) == 2
        assert b.args[1] == "a book to Mary"

    def test_said_type_reversal(self):
        g = Extraction(pred="said", args=["officials", "the war is over"])
        p = Extraction(pred="said", args=["the war is over", "officials"])
        score = binary_lenient_match(g, p)
        assert score[0] > 0.0  # should match via reversal

    def test_non_said_no_reversal(self):
        g = Extraction(pred="ate", args=["John", "an apple"])
        p = Extraction(pred="ate", args=["an apple", "John"])
        score = binary_lenient_match(g, p)
        # With wrong arg order and non-said verb, should still get partial
        # match since words overlap, just not a perfect match
        assert score[0] > 0.0 or score[1] > 0.0


class TestEvaluate:
    def test_perfect_predictions(self):
        gold = {
            "John is a teacher .": [Extraction(pred="is", args=["John", "a teacher"])],
        }
        pred = {
            "John is a teacher .": [
                Extraction(pred="is", args=["John", "a teacher"], confidence=0.9)
            ],
        }
        result = evaluate(gold, pred)
        assert result.f1 == 1.0
        assert result.precision == 1.0
        assert result.recall == 1.0
        assert result.auc > 0.0

    def test_no_predictions(self):
        gold = {
            "John is a teacher .": [Extraction(pred="is", args=["John", "a teacher"])],
        }
        pred: dict = {}
        result = evaluate(gold, pred)
        assert result.f1 == 0.0
        assert result.auc == 0.0

    def test_wrong_predictions(self):
        gold = {
            "John is a teacher .": [Extraction(pred="is", args=["John", "a teacher"])],
        }
        pred = {
            "John is a teacher .": [
                Extraction(pred="runs", args=["Mary", "a doctor"], confidence=0.9)
            ],
        }
        result = evaluate(gold, pred)
        assert result.f1 == 0.0

    def test_multiple_extractions(self):
        gold = {
            "John is a teacher and lives in London .": [
                Extraction(pred="is", args=["John", "a teacher"]),
                Extraction(pred="lives in", args=["John", "London"]),
            ],
        }
        pred = {
            "John is a teacher and lives in London .": [
                Extraction(pred="is", args=["John", "a teacher"], confidence=0.9),
                Extraction(pred="lives in", args=["John", "London"], confidence=0.8),
            ],
        }
        result = evaluate(gold, pred)
        assert result.f1 == 1.0


# ── Integration tests against original CaRB ────────────────────────────────


@pytest.mark.skipif(not HAS_CARB, reason="CaRB repo not found at ../CaRB")
class TestAgainstOriginalCarb:
    """Compare our implementation against the original CaRB on real data."""

    def _write_files(self, gold_content: str, pred_content: str) -> tuple[str, str]:
        gold_f = tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False)
        gold_f.write(gold_content)
        gold_f.close()
        pred_f = tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False)
        pred_f.write(pred_content)
        pred_f.close()
        return gold_f.name, pred_f.name

    def _cleanup(self, *paths):
        for p in paths:
            try:
                os.unlink(p)
            except OSError:
                pass

    def test_realistic_extractions(self):
        """Test with realistic, manually-crafted extractions."""
        gold_content = _make_gold_tsv(
            {
                "Barack Obama was born in Hawaii .": [
                    ("was born in", ["Barack Obama", "Hawaii"]),
                ],
                "The cat sat on the mat .": [
                    ("sat on", ["The cat", "the mat"]),
                ],
                "She said the project is complete .": [
                    ("said", ["She", "the project is complete"]),
                ],
                "John gave Mary a book at the library .": [
                    ("gave", ["John", "Mary", "a book", "at the library"]),
                ],
                "The company was founded in 1990 and is based in New York .": [
                    ("was founded in", ["The company", "1990"]),
                    ("is based in", ["The company", "New York"]),
                ],
            }
        )
        pred_content = _make_pred_tsv(
            {
                "Barack Obama was born in Hawaii .": [
                    (0.95, "was born in", ["Barack Obama", "Hawaii"]),
                ],
                "The cat sat on the mat .": [
                    (0.85, "sat on", ["The cat", "the mat"]),
                ],
                "She said the project is complete .": [
                    (0.80, "said", ["She", "the project is complete"]),
                ],
                "John gave Mary a book at the library .": [
                    (0.75, "gave", ["John", "Mary", "a book", "at the library"]),
                ],
                "The company was founded in 1990 and is based in New York .": [
                    (0.90, "was founded in", ["The company", "1990"]),
                    (0.70, "is based in", ["The company", "New York"]),
                ],
            }
        )
        gold_path, pred_path = self._write_files(gold_content, pred_content)
        try:
            orig_auc, orig_opt = _run_original_carb(gold_path, pred_path)
            ours = _run_our_carb(gold_path, pred_path)

            assert abs(ours.auc - orig_auc) < 0.01, (
                f"AUC mismatch: ours={ours.auc}, orig={orig_auc}"
            )
            assert abs(ours.f1 - orig_opt[2]) < 0.01, (
                f"F1 mismatch: ours={ours.f1}, orig={orig_opt[2]}"
            )
            assert abs(ours.precision - orig_opt[0]) < 0.01, (
                f"P mismatch: ours={ours.precision}, orig={orig_opt[0]}"
            )
            assert abs(ours.recall - orig_opt[1]) < 0.01, (
                f"R mismatch: ours={ours.recall}, orig={orig_opt[1]}"
            )
        finally:
            self._cleanup(gold_path, pred_path)

    def test_partial_matches(self):
        """Test with predictions that partially overlap gold."""
        gold_content = _make_gold_tsv(
            {
                "Albert Einstein developed the theory of relativity .": [
                    (
                        "developed",
                        ["Albert Einstein", "the theory of relativity"],
                    ),
                ],
                "The river flows through the ancient city .": [
                    ("flows through", ["The river", "the ancient city"]),
                ],
            }
        )
        pred_content = _make_pred_tsv(
            {
                "Albert Einstein developed the theory of relativity .": [
                    (0.9, "developed", ["Einstein", "the theory"]),
                ],
                "The river flows through the ancient city .": [
                    (0.8, "flows", ["The river", "the city"]),
                ],
            }
        )
        gold_path, pred_path = self._write_files(gold_content, pred_content)
        try:
            orig_auc, orig_opt = _run_original_carb(gold_path, pred_path)
            ours = _run_our_carb(gold_path, pred_path)

            assert abs(ours.auc - orig_auc) < 0.01, (
                f"AUC mismatch: ours={ours.auc}, orig={orig_auc}"
            )
            assert abs(ours.f1 - orig_opt[2]) < 0.01, (
                f"F1 mismatch: ours={ours.f1}, orig={orig_opt[2]}"
            )
        finally:
            self._cleanup(gold_path, pred_path)

    def test_random_extractions(self):
        """
        Generate many random extractions and verify our metrics agree
        with the original CaRB.
        """
        rng = np.random.default_rng(42)

        # Use a fixed vocabulary for reproducible random extractions
        vocab = [
            "the",
            "a",
            "is",
            "was",
            "in",
            "on",
            "at",
            "of",
            "John",
            "Mary",
            "cat",
            "dog",
            "house",
            "car",
            "city",
            "big",
            "small",
            "old",
            "new",
            "red",
            "blue",
            "green",
            "runs",
            "eats",
            "sees",
            "gives",
            "takes",
            "found",
            "built",
            "made",
            "born",
            "lives",
            "said",
            "told",
        ]
        sentences = [
            "John is a teacher in the big city .",
            "Mary was born in the old house .",
            "The cat runs on the red car .",
            "She said the dog eats a small blue fish .",
            "He found the old green car at the new city .",
            "The big dog lives in a small house .",
            "Mary gives John a red book .",
            "The old man sees a new blue car .",
        ]

        def random_phrase(n_words: int) -> str:
            return " ".join(rng.choice(vocab, size=n_words).tolist())

        # Build gold: 1-3 extractions per sentence
        gold_dict: dict[str, list[tuple[str, list[str]]]] = {}
        for sent in sentences:
            n_ex = rng.integers(1, 4)
            exs = []
            for _ in range(n_ex):
                pred = random_phrase(rng.integers(1, 3))
                arg1 = random_phrase(rng.integers(1, 4))
                arg2 = random_phrase(rng.integers(1, 4))
                exs.append((pred, [arg1, arg2]))
            gold_dict[sent] = exs

        # Build predicted: some correct, some partial, some wrong
        pred_dict: dict[str, list[tuple[float, str, list[str]]]] = {}
        for sent in sentences:
            exs = []
            gold_exs = gold_dict[sent]
            for pred_str, args in gold_exs:
                roll = rng.random()
                if roll < 0.3:
                    # Perfect copy
                    conf = float(rng.uniform(0.5, 1.0))
                    exs.append((conf, pred_str, list(args)))
                elif roll < 0.6:
                    # Partial: drop some words
                    conf = float(rng.uniform(0.3, 0.8))
                    partial_args = [
                        " ".join(a.split()[: max(1, len(a.split()) // 2)]) for a in args
                    ]
                    exs.append((conf, pred_str, partial_args))
                elif roll < 0.8:
                    # Wrong extraction
                    conf = float(rng.uniform(0.1, 0.5))
                    exs.append(
                        (
                            conf,
                            random_phrase(rng.integers(1, 3)),
                            [
                                random_phrase(rng.integers(1, 3)),
                                random_phrase(rng.integers(1, 3)),
                            ],
                        )
                    )
                # else: skip (no prediction for this gold)

            # Add some extra spurious predictions
            if rng.random() < 0.4:
                conf = float(rng.uniform(0.1, 0.4))
                exs.append(
                    (
                        conf,
                        random_phrase(rng.integers(1, 2)),
                        [
                            random_phrase(rng.integers(1, 3)),
                            random_phrase(rng.integers(1, 3)),
                        ],
                    )
                )

            if exs:
                pred_dict[sent] = exs

        gold_content = _make_gold_tsv(gold_dict)
        pred_content = _make_pred_tsv(pred_dict)
        gold_path, pred_path = self._write_files(gold_content, pred_content)

        try:
            orig_auc, orig_opt = _run_original_carb(gold_path, pred_path)
            ours = _run_our_carb(gold_path, pred_path)

            assert abs(ours.auc - orig_auc) < 0.02, (
                f"AUC mismatch: ours={ours.auc}, orig={orig_auc}"
            )
            assert abs(ours.f1 - orig_opt[2]) < 0.02, (
                f"F1 mismatch: ours={ours.f1}, orig={orig_opt[2]}"
            )
        finally:
            self._cleanup(gold_path, pred_path)

    def test_on_real_carb_data_subset(self):
        """
        Run on a subset of real CaRB gold data with synthetic predictions.
        """
        # Read first 20 sentences from gold
        from diffopenie.evaluation.carb_metrics import load_gold_file

        gold = load_gold_file(str(CARB_GOLD))
        subset_sents = list(gold.keys())[:20]
        gold_subset = {s: gold[s] for s in subset_sents}

        rng = np.random.default_rng(123)

        # Generate predictions: copy some gold with noise
        gold_lines = []
        pred_lines = []
        for sent in subset_sents:
            for ex in gold_subset[sent]:
                gold_lines.append("\t".join([sent, ex.pred] + ex.args))
                # Sometimes produce exact match, sometimes noisy
                conf = float(rng.uniform(0.3, 1.0))
                if rng.random() < 0.5:
                    # Exact copy
                    pred_lines.append("\t".join([sent, str(conf), ex.pred] + ex.args))
                else:
                    # Drop last word from each arg
                    noisy_args = [
                        " ".join(a.split()[:-1]) if len(a.split()) > 1 else a
                        for a in ex.args
                    ]
                    pred_lines.append(
                        "\t".join([sent, str(conf), ex.pred] + noisy_args)
                    )

        gold_content = "\n".join(gold_lines) + "\n"
        pred_content = "\n".join(pred_lines) + "\n"
        gold_path, pred_path = self._write_files(gold_content, pred_content)

        try:
            orig_auc, orig_opt = _run_original_carb(gold_path, pred_path)
            ours = _run_our_carb(gold_path, pred_path)

            assert abs(ours.auc - orig_auc) < 0.02, (
                f"AUC mismatch: ours={ours.auc}, orig={orig_auc}"
            )
            assert abs(ours.f1 - orig_opt[2]) < 0.02, (
                f"F1 mismatch: ours={ours.f1}, orig={orig_opt[2]}"
            )
            assert abs(ours.precision - orig_opt[0]) < 0.02, (
                f"P mismatch: ours={ours.precision}, orig={orig_opt[0]}"
            )
            assert abs(ours.recall - orig_opt[1]) < 0.02, (
                f"R mismatch: ours={ours.recall}, orig={orig_opt[1]}"
            )
        finally:
            self._cleanup(gold_path, pred_path)

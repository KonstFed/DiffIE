"""
CaRB-style evaluation metrics for Open IE.

Reimplements the core CaRB benchmark (F1, AUC) in a concise, self-contained
manner using only stdlib + numpy + sklearn.

The key algorithm:
  1. For each (gold, predicted) extraction pair, compute a word-overlap
     precision/recall score (the "binary lenient tuple match").
  2. For each confidence threshold, greedily match predicted→gold to maximize
     precision, accumulate P/R across all sentences.
  3. Compute optimal F1 and AUC over the resulting PR curve.
"""

from __future__ import annotations

import re
import string
from collections import defaultdict
from copy import copy
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import auc as sk_auc


# ── Data types ──────────────────────────────────────────────────────────────


@dataclass
class Extraction:
    """A single OIE extraction: predicate + arguments + confidence."""

    pred: str
    args: list[str] = field(default_factory=list)
    confidence: float = 1.0

    @staticmethod
    def from_gold_line(line: str) -> tuple[str, "Extraction"]:
        """Parse gold TSV line: sent\\tpred\\targ1\\targ2..."""
        parts = line.strip().split("\t")
        sent, pred = parts[0], parts[1]
        args = [a for a in parts[2:] if not a.startswith("C: ")]
        return sent, Extraction(pred=pred.strip(), args=[a.strip() for a in args])

    @staticmethod
    def from_tab_line(line: str) -> tuple[str, "Extraction"]:
        """Parse predicted TSV line: sent\\tconfidence\\tpred\\targ1\\targ2..."""
        parts = line.strip().split("\t")
        sent, conf, pred = parts[0], float(parts[1]), parts[2]
        args = list(parts[3:])
        return sent, Extraction(pred=pred, args=args, confidence=conf)


# ── Sentence normalization (matches CaRB) ───────────────────────────────────

_PTB_ESCAPES = [
    ("(", "-LRB-"),
    (")", "-RRB-"),
    ("[", "-LSB-"),
    ("]", "-RSB-"),
    ("{", "-LCB-"),
    ("}", "-RCB-"),
]
_PUNCT_RE = re.compile("[%s]" % re.escape(string.punctuation))


def _normalize_key(s: str) -> str:
    s = s.replace(" ", "")
    for u, e in _PTB_ESCAPES:
        s = s.replace(e, u)
    return _PUNCT_RE.sub("", s)


# ── Matching ────────────────────────────────────────────────────────────────

_FORMS_OF_BE = {"be", "is", "am", "are", "was", "were", "been", "being"}
_SAID_TYPE = {"said", "told", "added", "adds", "says"}


def _word_match_score(
    gold_words: list[str], pred_words: list[str]
) -> tuple[int, int, int]:
    """Returns (matching, len_gold, len_pred) with greedy word matching."""
    pred_remaining = list(pred_words)
    matching = 0
    for w in gold_words:
        if w in pred_remaining:
            matching += 1
            pred_remaining.remove(w)
    return matching, len(gold_words), len(pred_words)


def _lenient_tuple_match(gold: Extraction, pred: Extraction) -> tuple[float, float]:
    """
    Lenient tuple match: word-level overlap returning (precision, recall).
    Handles forms-of-be matching in predicates.
    """
    prec_num, prec_den = 0, 0
    rec_num, rec_den = 0, 0

    # Predicate matching
    g_words = gold.pred.lower().split()
    p_words = pred.pred.lower().split()
    prec_den += len(p_words)
    rec_den += len(g_words)

    p_remaining = list(p_words)
    matching = 0
    for w in g_words:
        if w in p_remaining:
            matching += 1
            p_remaining.remove(w)

    # "be" form matching
    if "be" in p_remaining:
        for form in _FORMS_OF_BE:
            if form in g_words:
                matching += 1
                p_remaining.remove("be")
                break

    if matching == 0:
        return 0.0, 0.0

    prec_num += matching
    rec_num += matching

    # Argument matching
    for i, g_arg in enumerate(gold.args):
        g_words = g_arg.lower().split()
        rec_den += len(g_words)
        if i >= len(pred.args):
            if i < 2:
                return 0.0, 0.0
            continue
        p_words = pred.args[i].lower().split()
        prec_den += len(p_words)
        p_remaining = list(p_words)
        matching = 0
        for w in g_words:
            if w in p_remaining:
                matching += 1
                p_remaining.remove(w)
        prec_num += matching
        rec_num += matching

    prec = prec_num / prec_den if prec_den > 0 else 0.0
    rec = rec_num / rec_den if rec_den > 0 else 0.0
    return prec, rec


def _binarize_extraction(ex: Extraction) -> Extraction:
    """Collapse args[1:] into a single second argument."""
    if len(ex.args) >= 2:
        out = copy(ex)
        out.args = [ex.args[0], " ".join(ex.args[1:])]
        return out
    return ex


def binary_lenient_match(gold: Extraction, pred: Extraction) -> tuple[float, float]:
    """
    Default CaRB matching: binarize then lenient tuple match.
    For "said"-type predicates, also try reversed argument order.
    """
    g = _binarize_extraction(gold)
    p = _binarize_extraction(pred)
    score = _lenient_tuple_match(g, p)

    # For said-type relations, also try reversed args
    if any(v in gold.pred.lower() for v in _SAID_TYPE) and len(pred.args) >= 2:
        p_rev = copy(p)
        p_rev.args = list(reversed(p.args))
        rev_score = _lenient_tuple_match(g, p_rev)
        score = max(score, rev_score)

    return score


# ── Benchmark evaluation ───────────────────────────────────────────────────

ExtractionDict = dict[str, list[Extraction]]


def load_gold_file(path: str) -> ExtractionDict:
    """Load gold extractions from CaRB gold TSV file."""
    d: ExtractionDict = defaultdict(list)
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            sent, ex = Extraction.from_gold_line(line)
            d[sent.strip()].append(ex)
    return dict(d)


def load_predicted_file(path: str) -> ExtractionDict:
    """Load predicted extractions from tab-separated file."""
    d: ExtractionDict = defaultdict(list)
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            sent, ex = Extraction.from_tab_line(line)
            d[sent].append(ex)
    return dict(d)


@dataclass
class CarbResult:
    """Result of CaRB evaluation."""

    auc: float
    precision: float
    recall: float
    f1: float

    def to_dict(self, prefix: str = "") -> dict[str, float]:
        return {
            f"{prefix}carb_auc": self.auc,
            f"{prefix}carb_precision": self.precision,
            f"{prefix}carb_recall": self.recall,
            f"{prefix}carb_f1": self.f1,
        }


def evaluate(
    gold: ExtractionDict,
    predicted: ExtractionDict,
    match_fn: callable = binary_lenient_match,
) -> CarbResult:
    """
    Run CaRB evaluation: compute AUC and optimal F1.

    Args:
        gold: dict mapping normalized sentence → list of gold extractions
        predicted: dict mapping normalized sentence → list of predicted extractions
        match_fn: function(gold_ex, pred_ex) → (precision, recall) score pair
    """
    # Normalize sentence keys
    gold_norm = {_normalize_key(k): v for k, v in gold.items()}
    pred_norm = {_normalize_key(k): v for k, v in predicted.items()}

    # Collect all confidence thresholds
    conf_thresholds = sorted(
        {ex.confidence for exs in pred_norm.values() for ex in exs}
    )

    if not conf_thresholds:
        return CarbResult(auc=0.0, precision=0.0, recall=0.0, f1=0.0)

    num_conf = len(conf_thresholds)
    p = np.zeros(num_conf)
    pl = np.zeros(num_conf)
    r = np.zeros(num_conf)
    rl = np.zeros(num_conf)

    for sent, gold_exs in gold_norm.items():
        pred_exs = pred_norm.get(sent, [])

        # Build score matrix
        scores = [[match_fn(g, p_ex) for p_ex in pred_exs] for g in gold_exs]

        # Process by ascending confidence threshold within this sentence
        sent_confs = sorted({ex.confidence for ex in pred_exs})
        prev_c = 0

        for conf in sent_confs:
            c = conf_thresholds.index(conf)
            ext_indices = [j for j, ex in enumerate(pred_exs) if ex.confidence >= conf]

            # Recall: sum of max recall per gold extraction
            recall_num = 0.0
            for row in scores:
                max_rec = max((row[j][1] for j in ext_indices), default=0.0)
                recall_num += max_rec

            # Precision: greedy match maximizing precision
            precision_num = 0.0
            selected_rows: set[int] = set()
            selected_cols: set[int] = set()
            num_matches = min(len(scores), len(ext_indices))

            for _ in range(num_matches):
                best_prec = -1.0
                best_i, best_j = -1, -1
                for i in range(len(scores)):
                    if i in selected_rows:
                        continue
                    for j in ext_indices:
                        if j in selected_cols:
                            continue
                        if scores[i][j][0] > best_prec:
                            best_prec = scores[i][j][0]
                            best_i, best_j = i, j
                if best_i >= 0:
                    selected_rows.add(best_i)
                    selected_cols.add(best_j)
                    precision_num += scores[best_i][best_j][0]

            p[prev_c : c + 1] += precision_num
            pl[prev_c : c + 1] += len(ext_indices)
            r[prev_c : c + 1] += recall_num
            rl[prev_c : c + 1] += len(scores)

            prev_c = c + 1

        # For thresholds above max sentence confidence, add gold count to recall denom
        rl[prev_c:] += len(scores)

    # Compute P/R/F1 at each threshold
    prec_scores = [a / b if b > 0 else 1.0 for a, b in zip(p, pl)]
    rec_scores = [a / b if b > 0 else 0.0 for a, b in zip(r, rl)]
    f1_scores = [
        2 * pr * rc / (pr + rc) if (pr + rc) > 0 else 0.0
        for pr, rc in zip(prec_scores, rec_scores)
    ]

    # Optimal F1
    if f1_scores:
        opt_idx = int(np.nanargmax(f1_scores))
        opt_p, opt_r, opt_f1 = (
            prec_scores[opt_idx],
            rec_scores[opt_idx],
            f1_scores[opt_idx],
        )
    else:
        opt_p, opt_r, opt_f1 = 0.0, 0.0, 0.0

    # AUC: add (recall=0, precision=1) anchor point
    rec_for_auc = rec_scores + [0.0]
    prec_for_auc = prec_scores + [1.0]
    auc_score = float(sk_auc(rec_for_auc, prec_for_auc))

    return CarbResult(
        auc=round(auc_score, 3),
        precision=round(opt_p, 3),
        recall=round(opt_r, 3),
        f1=round(opt_f1, 3),
    )

"""Retrieval + generation metrics. Pure Python (stdlib only) — cheap to test.

These are the reference implementations used by the pipeline.
"""
import re
import string
from collections import Counter
from typing import List

from . import config


def recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """Fraction of relevant docs found in the top-k.

    With a single gold doc (SQuAD v2) this is 1.0 if the gold paragraph is in the
    top-k, else 0.0 — i.e. hit@k.
    """
    if not relevant_ids:
        return 0.0
    hits = len(set(retrieved_ids[:k]) & set(relevant_ids))
    return hits / len(relevant_ids)


def mrr_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """Reciprocal rank of the first relevant doc within the top-k (0 if none)."""
    relevant = set(relevant_ids)
    for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def token_f1(prediction: str, ground_truth: str) -> float:
    """LEGACY (v2) token-overlap F1: no normalisation, set overlap.

    Kept only so the v2 numbers can be reproduced (F1_MODE=legacy). It is biased
    against longer answers: punctuation is not stripped ("Paris." != "Paris") and
    articles count as tokens. New results use squad_f1 via answer_f1().
    """
    pred = prediction.lower().split()
    gold = ground_truth.lower().split()
    if not pred or not gold:
        return 0.0
    common = set(pred) & set(gold)
    if not common:
        return 0.0
    precision = len(common) / len(pred)
    recall = len(common) / len(gold)
    return 2 * precision * recall / (precision + recall)


# ── SQuAD-standard answer metrics (official v2.0 evaluation script) ──
_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCT = set(string.punctuation)


def normalize_answer(s: str) -> str:
    """Lower-case, drop punctuation and articles, collapse whitespace (SQuAD)."""
    s = (s or "").lower()
    s = "".join(ch for ch in s if ch not in _PUNCT)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def squad_f1(prediction: str, ground_truth: str) -> float:
    """SQuAD F1: token overlap counted with multiplicity, after normalize_answer."""
    pred, gold = normalize_answer(prediction).split(), normalize_answer(ground_truth).split()
    if not pred or not gold:
        return float(pred == gold)
    common = Counter(pred) & Counter(gold)
    n = sum(common.values())
    if n == 0:
        return 0.0
    precision, recall = n / len(pred), n / len(gold)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def contains_gold(prediction: str, ground_truth: str) -> float:
    """1 if the normalised gold answer appears inside the normalised prediction.
    Lenient correctness check that does not punish full-sentence answers."""
    g = normalize_answer(ground_truth)
    return float(bool(g) and g in normalize_answer(prediction))


def answer_f1(prediction: str, ground_truth: str) -> float:
    """The F1 used by the pipeline: SQuAD F1 by default, legacy only to reproduce v2."""
    if config.F1_MODE == "legacy":
        return token_f1(prediction, ground_truth)
    return squad_f1(prediction, ground_truth)


def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0

"""Claim-Grounding Verifier (the project's novelty #1).

Idea: decompose an answer into atomic claims, then for each claim check whether
it is *entailed* by the retrieved evidence. An NLI cross-encoder judges each
(claim, passage) pair. Crucially we score each passage **individually** and take
the max — never concatenate all passages — so we stay under the model's 512-token
limit (FActScore / SummaC style).

    grounding score = (# supported claims) / (# claims)
    hallucination   = 1 - grounding score

Scorers are pluggable: MockScorer (lexical, 0-download) for smoke tests,
NLICrossEncoderScorer (transformers, GPU-accelerated) for the real run.
"""
import re
from typing import List, Tuple

from . import config


# ──────────────────────────────────────────────
# Claim handling
# ──────────────────────────────────────────────
def split_into_claims(answer: str) -> List[str]:
    """Split an answer into atomic claims (one per sentence). Drops empties."""
    parts = re.split(r"(?<=[.!?])\s+", answer.strip())
    return [p.strip() for p in parts if p.strip()]


# Leading framing phrases that bolt an un-entailed clause onto an otherwise
# grounded answer (e.g. "Based on the context, Paris is the capital."). We strip
# the lead-in so the NLI judge scores the real claim, not the boilerplate — this
# is what was dragging many correct short answers down to grounding 0.
_FRAMING_RE = re.compile(
    r"^\s*(based on (the )?(context|passage|text|document|evidence)s?|"
    r"according to (the )?(context|passage|text|document)s?|"
    r"as (stated|mentioned|described|noted) (in|by) (the )?(context|passage|text)|"
    r"the (context|passage|text|document)s? (states?|says?|mentions?|indicates?|shows?)( that)?|"
    r"the answer (to .*? )?is|in (the )?(given )?(context|passage|text))\b[\s,:\-]*",
    re.IGNORECASE,
)


def _strip_framing(claim: str) -> str:
    """Drop a leading framing phrase; keep the original if that empties the claim."""
    stripped = _FRAMING_RE.sub("", claim).strip()
    return stripped or claim


def _norm_tokens(text: str) -> List[str]:
    return [w.lower().strip(".,!?;:'\"()[]") for w in (text or "").split() if w.strip()]


def _quote_match(quote: str, passages: List[str]) -> float:
    """Max fraction of the cited quote's tokens found in any single passage, in [0,1].

    A cheap, verifiable grounding check for the cite variant: did the model quote
    something that actually appears in the retrieved context? (Catches fabricated
    citations and distractor-driven answers that can't point to a real sentence.)
    """
    qt = set(t for t in _norm_tokens(quote) if t)
    if not qt:
        return 0.0
    best = 0.0
    for p in passages:
        pt = set(_norm_tokens(p))
        if pt:
            best = max(best, len(qt & pt) / len(qt))
    return best


def grounding_score(claim_supported: List[bool]) -> float:
    """Fraction of claims supported by the evidence (0.0 if there are no claims)."""
    if not claim_supported:
        return 0.0
    return sum(1 for s in claim_supported if s) / len(claim_supported)


# ──────────────────────────────────────────────
# Scorers
# ──────────────────────────────────────────────
class MockScorer:
    """Lexical-overlap stand-in for NLI: entailment ~ fraction of claim content
    words present in the passage. Zero downloads — for offline smoke tests."""

    def score_pairs(self, pairs: List[Tuple[str, str]]) -> List[float]:
        out = []
        for claim, passage in pairs:
            claim_words = set(w.lower().strip(".,!?") for w in claim.split())
            claim_words -= {"the", "a", "an", "is", "of", "in", "to", "and", "i", "don't", "know"}
            if not claim_words:
                out.append(0.0)
                continue
            passage_words = set(w.lower().strip(".,!?") for w in passage.split())
            overlap = len(claim_words & passage_words) / len(claim_words)
            out.append(min(1.0, overlap))
        return out


class NLICrossEncoderScorer:
    """Real NLI judge via a cross-encoder. Premise=passage, hypothesis=claim;
    returns P(entailment). Loads lazily; uses GPU if available."""

    def __init__(self, model_name: str = config.NLI_MODEL, device: str = None):
        from sentence_transformers import CrossEncoder
        self.device = device or config.get_device()
        self.model = CrossEncoder(model_name, device=self.device)
        # cross-encoder/nli-* label order is [contradiction, entailment, neutral]
        self.entail_idx = 1

    def score_pairs(self, pairs: List[Tuple[str, str]]) -> List[float]:
        import numpy as np
        if not pairs:
            return []
        model_input = [(passage, claim) for claim, passage in pairs]  # (premise, hypothesis)
        logits = self.model.predict(model_input, convert_to_numpy=True)
        logits = np.atleast_2d(logits)
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exp / exp.sum(axis=1, keepdims=True)
        return probs[:, self.entail_idx].tolist()


# ──────────────────────────────────────────────
# Verifier
# ──────────────────────────────────────────────
class ClaimGroundingVerifier:
    def __init__(self, scorer=None, threshold: float = config.GROUNDING_THRESHOLD):
        self.scorer = scorer if scorer is not None else NLICrossEncoderScorer()
        self.threshold = threshold

    def verify(self, answer: str, passages: List[str],
               question: str = None, evidence_quote: str = None) -> dict:
        claims = [_strip_framing(c) for c in split_into_claims(answer)]
        quote_match = _quote_match(evidence_quote, passages) if evidence_quote else 0.0
        if not claims or not passages:
            return {"n_claims": len(claims), "claim_max_entail": [],
                    "supported": [], "grounding_score": 0.0,
                    "grounding_mean": 0.0, "grounding_max": 0.0,
                    "grounding_qa_mean": 0.0, "grounding_qa_max": 0.0,
                    "quote_match": quote_match, "hallucination_rate": 1.0}

        n_p = len(passages)

        def _claim_maxes(hyps: List[str]) -> List[float]:
            """Max entailment prob over passages, per hypothesis."""
            probs = self.scorer.score_pairs([(h, p) for h in hyps for p in passages])
            out = []
            for i in range(len(hyps)):
                window = probs[i * n_p:(i + 1) * n_p]
                out.append(max(window) if window else 0.0)
            return out

        # Answer-only grounding: does the evidence entail each answer claim?
        claim_max = _claim_maxes(claims)
        supported = [mx >= self.threshold for mx in claim_max]
        gs = grounding_score(supported)
        cont_mean = sum(claim_max) / len(claim_max)
        cont_max = max(claim_max)

        # Question-aware grounding: does the evidence entail THIS question's claim?
        # Targets distractors that entail a generic claim but don't answer the question.
        if question:
            qa_max = _claim_maxes([f"{question} {c}" for c in claims])
            qa_mean, qa_peak = sum(qa_max) / len(qa_max), max(qa_max)
        else:
            qa_mean, qa_peak = cont_mean, cont_max

        return {"n_claims": len(claims), "claim_max_entail": claim_max,
                "supported": supported, "grounding_score": gs,
                "grounding_mean": cont_mean, "grounding_max": cont_max,
                "grounding_qa_mean": qa_mean, "grounding_qa_max": qa_peak,
                "quote_match": quote_match, "hallucination_rate": 1.0 - gs}


def default_verifier(mock: bool = False) -> ClaimGroundingVerifier:
    return ClaimGroundingVerifier(scorer=MockScorer() if mock else None)

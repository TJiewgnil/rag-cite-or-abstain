"""Abstention evaluation (the project's novelty #2).

Reframes generation quality as a *calibration* problem via a 2x2 matrix:

                      model ANSWERS          model ABSTAINS
    answerable        correct_answer         false_refusal   (FRR)
    unanswerable      false_answer  (FAR)    correct_abstention

False Answer Rate (FAR) — answering when the evidence does not support an answer —
is the headline reliability number; lower is better.
"""
from typing import Dict, List

ABSTAIN_PHRASES = [
    "i don't know", "i do not know", "cannot answer", "can't answer",
    "not in the context", "no answer", "insufficient", "unable to answer",
    "does not contain", "not mentioned", "not provided",
]

# The four outcome labels.
CORRECT_ANSWER = "correct_answer"
FALSE_REFUSAL = "false_refusal"
FALSE_ANSWER = "false_answer"
CORRECT_ABSTENTION = "correct_abstention"


def is_abstention(answer: str) -> bool:
    """True if the model refused to answer (matches any abstain phrase)."""
    low = answer.lower()
    return any(phrase in low for phrase in ABSTAIN_PHRASES)


def classify_outcome(is_impossible: bool, abstained: bool) -> str:
    """Map (answerability, abstained) to one of the four 2x2 outcomes."""
    if is_impossible:
        return CORRECT_ABSTENTION if abstained else FALSE_ANSWER
    else:
        return FALSE_REFUSAL if abstained else CORRECT_ANSWER


def build_confusion(records: List[dict]) -> Dict[str, int]:
    """Count the four outcomes over records of {is_impossible, abstained}."""
    counts = {CORRECT_ANSWER: 0, FALSE_REFUSAL: 0,
              FALSE_ANSWER: 0, CORRECT_ABSTENTION: 0}
    for r in records:
        counts[classify_outcome(r["is_impossible"], r["abstained"])] += 1
    return counts


def compute_rates(counts: Dict[str, int]) -> Dict[str, float]:
    """Derive FAR, correct-abstention rate, false-refusal rate from counts."""
    n_unans = counts[FALSE_ANSWER] + counts[CORRECT_ABSTENTION]
    n_ans = counts[CORRECT_ANSWER] + counts[FALSE_REFUSAL]
    return {
        "false_answer_rate": counts[FALSE_ANSWER] / n_unans if n_unans else 0.0,
        "correct_abstention_rate": counts[CORRECT_ABSTENTION] / n_unans if n_unans else 0.0,
        "false_refusal_rate": counts[FALSE_REFUSAL] / n_ans if n_ans else 0.0,
        "correct_answer_rate": counts[CORRECT_ANSWER] / n_ans if n_ans else 0.0,
        "n_answerable": n_ans,
        "n_unanswerable": n_unans,
    }

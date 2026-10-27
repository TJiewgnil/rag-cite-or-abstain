"""Data structures + SQuAD v2 loading + jsonl helpers.

Nothing here imports torch/faiss, so this module is cheap to import and easy to
unit-test. `datasets` is imported lazily inside the loader only.
"""
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from . import config


# ──────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────
@dataclass
class Document:
    id: str
    text: str
    metadata: Optional[dict] = None


@dataclass
class RetrievalResult:
    doc: Document
    score: float
    method: str          # "bm25" | "dense" | "hybrid"


# ──────────────────────────────────────────────
# jsonl helpers (one JSON object per line)
# ──────────────────────────────────────────────
def write_jsonl(path, rows: List[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_corpus(corpus: List[Document], path=config.CORPUS_PATH) -> None:
    write_jsonl(path, [{"id": d.id, "text": d.text, "metadata": d.metadata}
                       for d in corpus])


def load_corpus(path=config.CORPUS_PATH) -> List[Document]:
    return [Document(id=r["id"], text=r["text"], metadata=r.get("metadata"))
            for r in read_jsonl(path)]


# ──────────────────────────────────────────────
# SQuAD v2 → corpus + balanced query subset
# ──────────────────────────────────────────────
def build_corpus_and_queries(n_queries: int = 20,
                             corpus_size: int = 50,
                             query_split: str = "validation",
                             seed: int = 42):
    """Build a (corpus, queries) pair from SQuAD v2.

    - queries: a balanced sample of answerable + unanswerable questions.
      In SQuAD v2 an unanswerable question has an empty answers["text"] list.
      Each query records its gold paragraph id and `is_impossible`.
    - corpus: the gold paragraphs of the sampled queries (so Recall is defined),
      plus extra unique paragraphs as distractors until `corpus_size` is reached
      (pulled from the same split, then from train if needed).

    Returns (corpus: List[Document], queries: List[dict]).
    """
    from datasets import load_dataset

    ds = load_dataset(config.SQUAD_DATASET, split=query_split)

    answerable, unanswerable = [], []
    for row in ds:
        is_imp = len(row["answers"]["text"]) == 0
        item = {
            "qid": row["id"],
            "question": row["question"],
            "gold_answer": "" if is_imp else row["answers"]["text"][0],
            "context": row["context"],
            "is_impossible": is_imp,
        }
        (unanswerable if is_imp else answerable).append(item)

    rng = random.Random(seed)
    rng.shuffle(answerable)
    rng.shuffle(unanswerable)

    half = max(1, n_queries // 2)
    sampled = answerable[:half] + unanswerable[:half]
    rng.shuffle(sampled)

    # Build corpus — gold contexts of sampled queries first.
    contexts, corpus = {}, []

    def add_context(text: str) -> str:
        if text not in contexts:
            cid = f"ctx_{len(corpus)}"
            contexts[text] = cid
            corpus.append(Document(id=cid, text=text))
        return contexts[text]

    queries = []
    for it in sampled:
        gold_id = add_context(it["context"])
        queries.append({
            "qid": it["qid"],
            "question": it["question"],
            "gold_answer": it["gold_answer"],
            "gold_doc_id": gold_id,
            "is_impossible": it["is_impossible"],
        })

    # Distractors from the same split.
    for it in answerable + unanswerable:
        if len(corpus) >= corpus_size:
            break
        add_context(it["context"])

    # If still short of corpus_size, pull distractors from train.
    if len(corpus) < corpus_size:
        train = load_dataset(config.SQUAD_DATASET, split="train")
        for row in train:
            if len(corpus) >= corpus_size:
                break
            add_context(row["context"])

    return corpus, queries


# ──────────────────────────────────────────────
# Synthetic data — for offline smoke tests (no network, no downloads)
# ──────────────────────────────────────────────
def build_synthetic(n_queries: int = 6, corpus_size: int = 10):
    """Tiny hand-made corpus + queries mirroring SQuAD v2 shape.

    Half the queries are answerable (answer present in gold paragraph), half are
    unanswerable (gold paragraph is on-topic but lacks the answer). Lets the whole
    pipeline run end-to-end with zero external dependencies.
    """
    facts = [
        ("The Eiffel Tower is located in Paris and was completed in 1889.", "Where is the Eiffel Tower?", "Paris"),
        ("Mount Everest is the highest mountain on Earth at 8849 metres.", "How high is Mount Everest?", "8849 metres"),
        ("Water boils at 100 degrees Celsius at sea level pressure.", "At what temperature does water boil?", "100 degrees Celsius"),
        ("The Great Wall of China is over 20000 kilometres long.", "How long is the Great Wall?", "over 20000 kilometres"),
        ("Photosynthesis converts sunlight into chemical energy in plants.", "What does photosynthesis convert sunlight into?", "chemical energy"),
    ]
    distractors = [
        "The Amazon rainforest produces a large share of the world's oxygen.",
        "Jupiter is the largest planet in the Solar System.",
        "The human heart has four chambers.",
        "Shakespeare wrote many plays during the Elizabethan era.",
        "Honey never spoils if stored in a sealed container.",
    ]

    corpus, queries = [], []
    # Answerable queries: gold paragraph contains the answer.
    for i, (ctx, q, ans) in enumerate(facts):
        doc = Document(id=f"ctx_{i}", text=ctx)
        corpus.append(doc)
        queries.append({"qid": f"ans_{i}", "question": q, "gold_answer": ans,
                        "gold_doc_id": doc.id, "is_impossible": False})
    # Unanswerable queries: reuse an on-topic paragraph but ask something it can't answer.
    unans = [
        ("ctx_0", "Who designed the Eiffel Tower's elevators?"),
        ("ctx_1", "What year was Mount Everest first measured?"),
        ("ctx_2", "What is the boiling point of water on Mars?"),
    ]
    for i, (gold_id, q) in enumerate(unans):
        queries.append({"qid": f"unans_{i}", "question": q, "gold_answer": "",
                        "gold_doc_id": gold_id, "is_impossible": True})

    # Pad corpus with distractors up to corpus_size.
    for j, text in enumerate(distractors):
        if len(corpus) >= corpus_size:
            break
        corpus.append(Document(id=f"ctx_{len(corpus)}", text=text))

    queries = queries[:max(1, n_queries)]
    return corpus, queries

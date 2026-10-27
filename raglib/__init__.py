"""raglib — toolkit for studying when a RAG system should answer or abstain.

Modules (import lazily; heavy deps like torch/faiss load only when used):
    config      paths, model ids, hyperparameters
    data        Document/RetrievalResult, SQuAD v2 loader, jsonl helpers
    retrievers  BM25 / Dense / Hybrid retrievers (+ rrf_fuse)
    generator   GeminiGenerator (Vertex AI) + MockGenerator (0-cost smoke)
    grounding   ClaimGroundingVerifier (NLI) + MockScorer
    abstention  abstention detection, 2x2 confusion matrix, False Answer Rate
    metrics     recall@k / mrr@k / token_f1
"""

__all__ = [
    "config", "data", "retrievers", "generator",
    "grounding", "abstention", "metrics",
]

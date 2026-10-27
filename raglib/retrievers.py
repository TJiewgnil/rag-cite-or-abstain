"""Retrievers: BM25 (sparse) · Dense (embeddings+FAISS) · Hybrid (RRF).

Heavy deps (numpy / rank_bm25 / sentence-transformers / faiss) are imported
lazily inside the constructors, so `import raglib.retrievers` is cheap and the
pure `rrf_fuse` helper is unit-testable without any of them.
"""
from typing import Dict, List, Tuple

from .data import Document, RetrievalResult
from . import config


# ──────────────────────────────────────────────
# Reciprocal Rank Fusion — pure helper (no heavy deps)
# ──────────────────────────────────────────────
def rrf_fuse(rankings: List[List[str]], k: int = config.RRF_K,
             top_k: int = config.TOP_K) -> List[Tuple[str, float]]:
    """Fuse several ranked lists of doc ids into one.

    Reciprocal Rank Fusion: a doc at 0-based `rank` in a list contributes
    1 / (k + rank + 1) to its score; scores sum across lists. Returns the
    top_k (doc_id, score) pairs, highest first.
    """
    scores: Dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


# ──────────────────────────────────────────────
# BM25 — sparse lexical baseline
# ──────────────────────────────────────────────
class BM25Retriever:
    def __init__(self, documents: List[Document]):
        from rank_bm25 import BM25Okapi
        self.documents = documents
        self._id2doc = {d.id: d for d in documents}
        tokenized = [d.text.lower().split() for d in documents]
        self.bm25 = BM25Okapi(tokenized)

    def retrieve(self, query: str, top_k: int = config.TOP_K) -> List[RetrievalResult]:
        import numpy as np
        scores = self.bm25.get_scores(query.lower().split())
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [RetrievalResult(self.documents[i], float(scores[i]), "bm25")
                for i in top_idx]


# ──────────────────────────────────────────────
# Dense — sentence-transformers + FAISS (GPU-accelerated encoding)
# ──────────────────────────────────────────────
class DenseRetriever:
    def __init__(self, documents: List[Document],
                 model_name: str = config.DENSE_MODEL, device: str = None):
        import numpy as np
        import faiss
        from sentence_transformers import SentenceTransformer

        self.documents = documents
        self._id2doc = {d.id: d for d in documents}
        device = device or config.get_device()
        self.encoder = SentenceTransformer(model_name, device=device)
        # bge wants an instruction on the query side only; harmless "" otherwise.
        self._query_prefix = config.BGE_QUERY_INSTRUCTION if "bge" in model_name.lower() else ""

        emb = self.encoder.encode([d.text for d in documents],
                                  batch_size=64, convert_to_numpy=True,
                                  show_progress_bar=True)
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        self.index = faiss.IndexFlatIP(emb.shape[1])   # inner product = cosine (normalised)
        self.index.add(emb.astype(np.float32))

    def retrieve(self, query: str, top_k: int = config.TOP_K) -> List[RetrievalResult]:
        import numpy as np
        q = self.encoder.encode([self._query_prefix + query], convert_to_numpy=True)
        q = q / np.linalg.norm(q, axis=1, keepdims=True)
        scores, idx = self.index.search(q.astype(np.float32), top_k)
        return [RetrievalResult(self.documents[i], float(scores[0][r]), "dense")
                for r, i in enumerate(idx[0]) if i != -1]


# ──────────────────────────────────────────────
# Hybrid — Reciprocal Rank Fusion of BM25 + Dense
# ──────────────────────────────────────────────
class HybridRetriever:
    def __init__(self, bm25: BM25Retriever, dense: DenseRetriever,
                 k: int = config.RRF_K):
        self.bm25 = bm25
        self.dense = dense
        self.k = k
        self._id2doc = {**bm25._id2doc, **dense._id2doc}

    def retrieve(self, query: str, top_k: int = config.TOP_K) -> List[RetrievalResult]:
        bm25_ids = [r.doc.id for r in self.bm25.retrieve(query, top_k * 2)]
        dense_ids = [r.doc.id for r in self.dense.retrieve(query, top_k * 2)]
        fused = rrf_fuse([bm25_ids, dense_ids], k=self.k, top_k=top_k)
        return [RetrievalResult(self._id2doc[doc_id], score, "hybrid")
                for doc_id, score in fused]


def build_retrievers(documents: List[Document], include_dense: bool = True) -> dict:
    """Factory: build the requested retrievers. Set include_dense=False for
    offline/0-download smoke tests (BM25 only)."""
    bm25 = BM25Retriever(documents)
    retrievers = {"BM25": bm25}
    if include_dense:
        dense = DenseRetriever(documents)
        retrievers["Dense"] = dense
        retrievers["Hybrid"] = HybridRetriever(bm25, dense)
    return retrievers

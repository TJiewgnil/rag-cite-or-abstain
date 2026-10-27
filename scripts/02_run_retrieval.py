"""Step 02 — run BM25 / Dense / Hybrid retrieval and score Recall@k, MRR@k.

    python scripts/02_run_retrieval.py                 # all 3 retrievers
    python scripts/02_run_retrieval.py --no-dense      # BM25 only (offline)

Reads data/{corpus,queries}.jsonl. Writes outputs/retrieval.jsonl and prints metrics.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from raglib import config, data, metrics, retrievers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top_k", type=int, default=config.TOP_K)
    ap.add_argument("--no-dense", action="store_true", help="BM25 only (no model download)")
    args = ap.parse_args()

    corpus = data.load_corpus(config.CORPUS_PATH)
    queries = data.read_jsonl(config.QUERIES_PATH)
    retr = retrievers.build_retrievers(corpus, include_dense=not args.no_dense)

    rows = []
    for name, r in retr.items():
        for q in queries:
            results = r.retrieve(q["question"], top_k=args.top_k)
            rows.append({
                "qid": q["qid"], "method": name,
                "retrieved_doc_ids": [res.doc.id for res in results],
                "scores": [res.score for res in results],
                "gold_doc_id": q["gold_doc_id"],
                "is_impossible": q["is_impossible"],
            })
    data.write_jsonl(config.RETRIEVAL_PATH, rows)

    print(f"[02] wrote {len(rows)} rows -> {config.RETRIEVAL_PATH}")
    summary = {}
    for name in retr:
        mrows = [r for r in rows if r["method"] == name]
        rec = metrics.mean([metrics.recall_at_k(r["retrieved_doc_ids"], [r["gold_doc_id"]], args.top_k) for r in mrows])
        mrr = metrics.mean([metrics.mrr_at_k(r["retrieved_doc_ids"], [r["gold_doc_id"]], args.top_k) for r in mrows])
        summary[name] = {"recall@k": rec, "mrr@k": mrr}
        print(f"       {name:<7} Recall@{args.top_k}={rec:.3f}  MRR@{args.top_k}={mrr:.3f}")

    with open(config.OUTPUT_DIR / "metrics_retrieval.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()

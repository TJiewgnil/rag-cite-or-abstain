"""Step 04 — run the Claim-Grounding Verifier over the cached generations.

    python scripts/04_grounding.py --mock      # 0-cost lexical scorer (offline)
    python scripts/04_grounding.py             # real NLI cross-encoder (GPU)

Reads outputs/generations.jsonl + outputs/retrieval.jsonl + data/corpus.jsonl.
Writes outputs/grounding.jsonl (grounding score per generation).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from raglib import config, data, grounding


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true", help="use MockScorer (no download)")
    ap.add_argument("--top_k", type=int, default=config.TOP_K)
    args = ap.parse_args()

    generations = data.read_jsonl(config.GENERATIONS_PATH)
    corpus = {d.id: d.text for d in data.load_corpus(config.CORPUS_PATH)}
    retrieval = {(r["method"], r["qid"]): r for r in data.read_jsonl(config.RETRIEVAL_PATH)}
    qtext = {q["qid"]: q["question"] for q in data.read_jsonl(config.QUERIES_PATH)}

    verifier = grounding.default_verifier(mock=args.mock)
    print(f"[04] verifier scorer={'mock' if args.mock else config.NLI_MODEL}  "
          f"items={len(generations)}")

    rows = []
    for g in generations:
        rrow = retrieval.get((g["method"], g["qid"]))
        passages = [corpus[i] for i in rrow["retrieved_doc_ids"][:args.top_k] if i in corpus] if rrow else []
        res = verifier.verify(g["answer"], passages,
                              question=qtext.get(g["qid"]),
                              evidence_quote=g.get("evidence_quote"))
        rows.append({
            "key": g["key"], "method": g["method"], "variant": g["variant"],
            "qid": g["qid"], "grounding_score": res["grounding_score"],
            "grounding_mean": res["grounding_mean"], "grounding_max": res["grounding_max"],
            "grounding_qa_mean": res["grounding_qa_mean"], "grounding_qa_max": res["grounding_qa_max"],
            "quote_match": res["quote_match"],
            "claim_max_entail": res["claim_max_entail"],
            "hallucination_rate": res["hallucination_rate"], "n_claims": res["n_claims"],
        })
    data.write_jsonl(config.GROUNDING_PATH, rows)
    print(f"[04] wrote {len(rows)} rows -> {config.GROUNDING_PATH}")


if __name__ == "__main__":
    main()

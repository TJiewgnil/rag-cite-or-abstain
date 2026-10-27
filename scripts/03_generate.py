"""Step 03 — generate answers once and cache them (the only step that costs money).

    python scripts/03_generate.py --mock                      # 0-cost, offline
    python scripts/03_generate.py --concurrency 12            # real Gemini

Set RAG_OUTPUT_SUBDIR=outputs_v3 to write into a new results folder (see 03b).

Reads outputs/retrieval.jsonl + data/{corpus,queries}.jsonl.
Writes/append-caches outputs/generations.jsonl (keyed by method::variant::qid;
re-runs skip cached keys, so generation happens exactly once).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from raglib import config, data, generator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true", help="use MockGenerator (no API)")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--top_k", type=int, default=config.TOP_K)
    ap.add_argument("--methods", nargs="*", default=None, help="subset of methods")
    ap.add_argument("--variants", nargs="*", default=list(config.PROMPT_VARIANTS))
    ap.add_argument("--keys_file", default=None,
                    help="only generate these cache keys (one per line), e.g. the 30-item "
                         "cite diagnostic written by 03b_cite_rerun_prep.py --sample 30")
    args = ap.parse_args()
    keep = None
    if args.keys_file:
        keep = {l.strip() for l in open(args.keys_file, encoding="utf-8") if l.strip()}

    retrieval = data.read_jsonl(config.RETRIEVAL_PATH)
    corpus = {d.id: d.text for d in data.load_corpus(config.CORPUS_PATH)}
    qtext = {q["qid"]: q["question"] for q in data.read_jsonl(config.QUERIES_PATH)}

    tasks = []
    for r in retrieval:
        if args.methods and r["method"] not in args.methods:
            continue
        context_texts = [corpus[i] for i in r["retrieved_doc_ids"][:args.top_k] if i in corpus]
        for variant in args.variants:
            key = generator.cache_key(r["method"], variant, r["qid"])
            if keep is not None and key not in keep:
                continue
            tasks.append({
                "key": key,
                "method": r["method"], "variant": variant, "qid": r["qid"],
                "question": qtext[r["qid"]], "context_texts": context_texts,
            })

    gen = generator.MockGenerator() if args.mock else generator.GeminiGenerator()
    print(f"[03] generator={gen.model_name}  tasks={len(tasks)}  "
          f"concurrency={args.concurrency}")
    rows = generator.generate_batch(gen, tasks, config.GENERATIONS_PATH,
                                    concurrency=args.concurrency)
    print(f"[03] total cached generations={len(rows)} -> {config.GENERATIONS_PATH}")


if __name__ == "__main__":
    main()

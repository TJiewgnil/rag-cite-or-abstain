"""Step 01 — build the corpus + balanced query subset from SQuAD v2.

    python scripts/01_build_data.py --n_queries 20 --corpus_size 50   # smoke
    python scripts/01_build_data.py --n_queries 800 --corpus_size 20000  # full

Writes data/corpus.jsonl and data/queries.jsonl.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from raglib import config, data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_queries", type=int, default=20)
    ap.add_argument("--corpus_size", type=int, default=50)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--synthetic", action="store_true",
                    help="use offline synthetic data (no download)")
    args = ap.parse_args()

    if args.synthetic:
        corpus, queries = data.build_synthetic(args.n_queries, args.corpus_size)
    else:
        corpus, queries = data.build_corpus_and_queries(
            n_queries=args.n_queries, corpus_size=args.corpus_size,
            query_split=args.split, seed=args.seed)

    data.save_corpus(corpus, config.CORPUS_PATH)
    data.write_jsonl(config.QUERIES_PATH, queries)

    n_unans = sum(1 for q in queries if q["is_impossible"])
    print(f"[01] corpus={len(corpus)} docs -> {config.CORPUS_PATH}")
    print(f"[01] queries={len(queries)} ({n_unans} unanswerable / "
          f"{len(queries) - n_unans} answerable) -> {config.QUERIES_PATH}")


if __name__ == "__main__":
    main()

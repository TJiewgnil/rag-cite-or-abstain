"""Step 00 — the smoke-test ladder. Validate concepts cheaply before scaling up.

    python scripts/00_smoke_test.py --stage env        # S0: imports + GPU check
    python scripts/00_smoke_test.py --offline          # S1-S6: 0-cost end-to-end
                                                        #        (synthetic data,
                                                        #         BM25 + mock gen + mock NLI)

The offline run writes only to outputs/smoke/ and data/smoke_* so it never
pollutes the real generation cache. After it passes, do the real run on the VM:
    01_build_data -> 02_run_retrieval -> 03_generate -> 04_grounding -> 05_evaluate
"""
import argparse
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from raglib import config, data, evaluate, generator, grounding, metrics, retrievers


def stage_env():
    print("== S0: environment ==")
    print(f"  device = {config.get_device()}")
    for mod in ["numpy", "torch", "rank_bm25", "sentence_transformers",
                "faiss", "datasets", "transformers", "vertexai"]:
        try:
            importlib.import_module(mod)
            print(f"  [ok]      {mod}")
        except Exception as e:
            print(f"  [missing] {mod}  ({e.__class__.__name__})")
    print("S0 done (install any [missing] deps on the VM before the real run).")


def stage_offline(n_queries=8, corpus_size=10):
    print("== S1-S6: offline end-to-end (synthetic + BM25 + mock gen + mock NLI) ==")
    smoke = config.OUTPUT_DIR / "smoke"
    smoke.mkdir(exist_ok=True)
    corpus_path = config.DATA_DIR / "smoke_corpus.jsonl"
    queries_path = config.DATA_DIR / "smoke_queries.jsonl"
    gen_path = smoke / "generations.jsonl"
    if gen_path.exists():
        gen_path.unlink()   # deterministic (cache appends otherwise)

    # S1 — data
    corpus, queries = data.build_synthetic(n_queries, corpus_size)
    data.save_corpus(corpus, corpus_path)
    data.write_jsonl(queries_path, queries)
    assert corpus and any(q["is_impossible"] for q in queries) and any(not q["is_impossible"] for q in queries)
    print(f"  S1 data: corpus={len(corpus)}, queries={len(queries)} "
          f"({sum(q['is_impossible'] for q in queries)} unanswerable)  PASS")

    # S2 — retrieval (BM25 only, no downloads)
    retr = retrievers.build_retrievers(corpus, include_dense=False)
    rrows = []
    for name, r in retr.items():
        for q in queries:
            res = r.retrieve(q["question"], top_k=config.TOP_K)
            rrows.append({"qid": q["qid"], "method": name,
                          "retrieved_doc_ids": [x.doc.id for x in res],
                          "scores": [x.score for x in res],
                          "gold_doc_id": q["gold_doc_id"],
                          "is_impossible": q["is_impossible"]})
    assert all(rr["retrieved_doc_ids"] for rr in rrows)
    rec = metrics.mean([metrics.recall_at_k(rr["retrieved_doc_ids"], [rr["gold_doc_id"]], config.TOP_K) for rr in rrows])
    print(f"  S2 retrieval: BM25 Recall@{config.TOP_K}={rec:.3f}  PASS")

    # S3 — generation (mock)
    cmap = {d.id: d.text for d in corpus}
    tasks = []
    for rr in rrows:
        ctx = [cmap[i] for i in rr["retrieved_doc_ids"][:config.TOP_K]]
        for v in config.PROMPT_VARIANTS:
            tasks.append({"key": generator.cache_key(rr["method"], v, rr["qid"]),
                          "method": rr["method"], "variant": v, "qid": rr["qid"],
                          "question": next(q["question"] for q in queries if q["qid"] == rr["qid"]),
                          "context_texts": ctx})
    gens = generator.generate_batch(generator.MockGenerator(), tasks, gen_path)
    assert len(gens) == len(tasks)
    print(f"  S3 generation(mock): {len(gens)} answers cached  PASS")

    # S4 — grounding (mock NLI): entailed pair should outscore a contradiction
    verifier = grounding.default_verifier(mock=True)
    e = verifier.scorer.score_pairs([("The Eiffel Tower is in Paris.", "The Eiffel Tower is located in Paris and was completed in 1889.")])[0]
    c = verifier.scorer.score_pairs([("The Eiffel Tower is in Paris.", "The human heart has four chambers.")])[0]
    assert e > c
    rmap = {(x["method"], x["qid"]): x for x in rrows}
    grows = []
    for g in gens:
        rr = rmap[(g["method"], g["qid"])]
        passages = [cmap[i] for i in rr["retrieved_doc_ids"][:config.TOP_K]]
        res = verifier.verify(g["answer"], passages)
        grows.append({"key": g["key"], "method": g["method"], "variant": g["variant"],
                      "qid": g["qid"], "grounding_score": res["grounding_score"],
                      "hallucination_rate": res["hallucination_rate"], "n_claims": res["n_claims"]})
    print(f"  S4 grounding(mock): entailment={e:.2f} > contradiction={c:.2f}  PASS")

    # S5 — evaluate: 2x2 confusion must sum to #queries for every method|variant
    summary = evaluate.evaluate_all(rrows, gens, grows, queries, top_k=config.TOP_K)
    for mv, r in summary["by_method_variant"].items():
        assert sum(r["confusion"].values()) == len(queries), (mv, r["confusion"])
    print(f"  S5 evaluate: confusion sums == {len(queries)} for all method|variant  PASS")

    # S6 — end-to-end produced everything
    print("  S6 end-to-end: data+retrieval+generation+grounding+evaluation OK  PASS")

    mv0 = next(iter(summary["by_method_variant"]))
    print(f"\n  sample [{mv0}] rates = {summary['by_method_variant'][mv0]['rates']}")
    print("\nALL OFFLINE SMOKE STAGES PASSED")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["env", "offline"], default=None)
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()
    if args.stage == "env":
        stage_env()
    else:
        stage_offline()


if __name__ == "__main__":
    main()

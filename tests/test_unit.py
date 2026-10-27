"""Unit tests for raglib pure-logic functions (no network / no GPU / no pytest).

Run:   python tests/test_unit.py      (exits non-zero on any failure)
"""
import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from raglib import abstention, data, evaluate, generator, grounding, metrics, retrievers


def test_metrics():
    assert metrics.recall_at_k(["a", "b", "c"], ["b"], 5) == 1.0
    assert metrics.recall_at_k(["a", "b", "c"], ["z"], 5) == 0.0
    assert metrics.mrr_at_k(["a", "b", "c"], ["b"], 5) == 0.5
    assert metrics.mrr_at_k(["a", "b"], ["z"], 5) == 0.0
    assert math.isclose(metrics.token_f1("the cat sat", "the cat ran"), 2 / 3, rel_tol=1e-6)
    assert metrics.mean([]) == 0.0
    # SQuAD-standard metrics (review C2)
    assert metrics.normalize_answer("The Eiffel Tower, a landmark.") == "eiffel tower landmark"
    assert metrics.normalize_answer("Theatre and an anthem") == "theatre and anthem"
    assert metrics.squad_f1("Paris.", "Paris") == 1.0          # legacy token_f1 gives 0 here
    assert metrics.token_f1("Paris.", "Paris") == 0.0
    assert math.isclose(metrics.squad_f1("the cat cat", "cat"), 2 / 3)   # counted overlap
    assert metrics.exact_match("The Paris!", "paris") == 1.0
    assert metrics.contains_gold("It is in Paris, France.", "Paris") == 1.0
    assert metrics.contains_gold("anything", "") == 0.0


def test_rrf():
    out = retrievers.rrf_fuse([["a", "b", "c"], ["b", "c", "a"]], k=60, top_k=3)
    ids = [d for d, _ in out]
    assert ids[0] == "b"
    assert set(ids) == {"a", "b", "c"}
    assert out[0][1] >= out[1][1] >= out[2][1]


def test_grounding():
    assert grounding.split_into_claims("A. B! C?") == ["A.", "B!", "C?"]
    assert math.isclose(grounding.grounding_score([True, True, False]), 2 / 3, rel_tol=1e-6)
    assert grounding.grounding_score([]) == 0.0
    v = grounding.default_verifier(mock=True)
    res = v.verify("The Eiffel Tower is in Paris.",
                   ["The Eiffel Tower is located in Paris and opened in 1889."])
    assert res["grounding_score"] == 1.0
    res2 = v.verify("The Eiffel Tower is in Paris.", ["Jupiter is the largest planet."])
    assert res2["grounding_score"] == 0.0


def test_abstention():
    assert abstention.is_abstention("I don't know.") is True
    assert abstention.is_abstention("Paris.") is False
    assert abstention.classify_outcome(True, False) == abstention.FALSE_ANSWER
    assert abstention.classify_outcome(True, True) == abstention.CORRECT_ABSTENTION
    assert abstention.classify_outcome(False, False) == abstention.CORRECT_ANSWER
    assert abstention.classify_outcome(False, True) == abstention.FALSE_REFUSAL
    recs = [{"is_impossible": True, "abstained": False},
            {"is_impossible": True, "abstained": True}]
    counts = abstention.build_confusion(recs)
    rates = abstention.compute_rates(counts)
    assert rates["false_answer_rate"] == 0.5
    assert rates["correct_abstention_rate"] == 0.5


def test_data_jsonl_and_synthetic():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "x.jsonl"
        data.write_jsonl(p, [{"a": 1}, {"a": 2}])
        assert data.read_jsonl(p) == [{"a": 1}, {"a": 2}]
    corpus, queries = data.build_synthetic(n_queries=6, corpus_size=8)
    assert len(corpus) >= 5
    assert any(q["is_impossible"] for q in queries)
    assert all(q["gold_doc_id"] in {d.id for d in corpus} for q in queries)


def test_evaluate_end_to_end():
    corpus, queries = data.build_synthetic(n_queries=8, corpus_size=10)
    cmap = {d.id: d.text for d in corpus}
    bm25 = retrievers.BM25Retriever(corpus)
    rrows = []
    for q in queries:
        res = bm25.retrieve(q["question"], top_k=5)
        rrows.append({"qid": q["qid"], "method": "BM25",
                      "retrieved_doc_ids": [x.doc.id for x in res],
                      "scores": [x.score for x in res],
                      "gold_doc_id": q["gold_doc_id"], "is_impossible": q["is_impossible"]})
    tasks = []
    for rr in rrows:
        ctx = [cmap[i] for i in rr["retrieved_doc_ids"][:5]]
        for v in ("naive", "explicit"):
            tasks.append({"key": generator.cache_key("BM25", v, rr["qid"]),
                          "method": "BM25", "variant": v, "qid": rr["qid"],
                          "question": next(q["question"] for q in queries if q["qid"] == rr["qid"]),
                          "context_texts": ctx})
    with tempfile.TemporaryDirectory() as td:
        gens = generator.generate_batch(generator.MockGenerator(), tasks,
                                        Path(td) / "g.jsonl")
    verifier = grounding.default_verifier(mock=True)
    rmap = {(x["method"], x["qid"]): x for x in rrows}
    grows = []
    for g in gens:
        rr = rmap[(g["method"], g["qid"])]
        passages = [cmap[i] for i in rr["retrieved_doc_ids"][:5]]
        res = verifier.verify(g["answer"], passages)
        grows.append({"key": g["key"], "grounding_score": res["grounding_score"],
                      "hallucination_rate": res["hallucination_rate"], "n_claims": res["n_claims"],
                      "method": g["method"], "variant": g["variant"], "qid": g["qid"]})
    summary = evaluate.evaluate_all(rrows, gens, grows, queries, top_k=5)
    for mv, r in summary["by_method_variant"].items():
        assert sum(r["confusion"].values()) == len(queries)
    assert "BM25" in summary["retrieval"]


def test_cite_parsing():
    P = generator.parse_cite_response
    ok = P('{"has_answer": true, "evidence_quote": "Paris is big.", "answer": "Paris"}')
    assert ok["has_answer"] is True and ok["parse_status"] == "ok"
    # v2 bug: unparseable text used to become has_answer=True -> must now be a format failure
    for bad in ["", "Sure! The answer is Paris", '{"has_answer": true, "evid']:
        r = P(bad)
        assert r["has_answer"] is False and r["parse_status"] in generator.FORMAT_FAILURES, bad
    # string "false" must not be read as True
    r = P('{"has_answer": "false", "evidence_quote": "", "answer": ""}')
    assert r["has_answer"] is False and r["parse_status"] == "repaired"
    # JSON array, code fences, alternative key, extra text around the object
    assert P('[{"has_answer": true, "evidence_quote": "q", "answer": "a"}]')["has_answer"] is True
    fenced = '```json\n{"has_answer": false, "evidence_quote": "", "answer": ""}\n```'
    assert P(fenced)["parse_status"] == "ok"
    r = P('{"has_answer": true, "quote": "q here", "answer": "a"}')
    assert r["evidence_quote"] == "q here" and r["parse_status"] == "repaired"
    assert P('Here: {"has_answer": true, "evidence_quote": "q", "answer": "a"} done')["has_answer"] is True
    assert P('{"answer": "a"}')["parse_status"] == "invalid"      # no has_answer key


def test_cite_outcomes():
    ctx = ["The Eiffel Tower is located in Paris and was completed in 1889."]
    t = {"key": "BM25::cite::q1", "method": "BM25", "variant": "cite", "qid": "q1",
         "context_texts": ctx}
    def row(raw):
        res = generator.parse_cite_response(raw)
        res.update(raw_response=raw, finish_reason="STOP", attempts=1)
        return generator.cite_row(t, res, "test")
    r = row('{"has_answer": true, "evidence_quote": "The Eiffel Tower is located in Paris", "answer": "Paris"}')
    assert r["cite_outcome"] == "answered" and r["answer"] == "Paris" and r["answer_in_quote"]
    r = row('{"has_answer": false, "evidence_quote": "", "answer": ""}')
    assert r["cite_outcome"] == "model_declined" and r["answer"] == generator.ABSTAIN_TEXT
    r = row('{"has_answer": true, "evidence_quote": "Totally made up words here", "answer": "x"}')
    assert r["cite_outcome"] == "quote_unverified"
    r = row("not json at all")
    assert r["cite_outcome"] == "format_failure" and r["raw_response"] == "not json at all"
    # v2 rows (no cite_outcome field) are inferred
    assert generator.infer_cite_outcome({"answer": "I don't know.", "has_answer": True,
                                         "evidence_quote": ""}) == "format_failure_suspected"
    assert generator.infer_cite_outcome({"answer": "I don't know.", "has_answer": False}) == "model_declined"
    assert generator.infer_cite_outcome({"answer": "OPEC", "has_answer": True}) == "answered"
    # mock goes through the same parser
    m = generator.MockGenerator().generate_cite("Where is the tower?", ctx)
    assert m["parse_status"] == "ok" and "raw_response" in m


TESTS = [test_metrics, test_rrf, test_grounding, test_abstention,
         test_data_jsonl_and_synthetic, test_evaluate_end_to_end,
         test_cite_parsing, test_cite_outcomes]


def main():
    failed = 0
    for t in TESTS:
        try:
            t()
            print(f"  [PASS] {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"  [FAIL] {t.__name__} -> {e.__class__.__name__}: {e}")
    print(f"\n{len(TESTS) - failed}/{len(TESTS)} unit tests passed.")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

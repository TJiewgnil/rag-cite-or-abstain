"""Step 11 — where do false answers come from? (review C3 mechanism + manual audit). Free.

Needs the full 20k-paragraph data/corpus.jsonl (run on the VM, or copy it locally).

Part 1 — mechanism. The v2 paper explained "BM25·cite has lower FAR than Dense/Hybrid·cite"
by stronger retrievers fetching the designated (gold-like) paragraph more often. But BM25
and Dense fetch it almost equally often (106 vs 108 of 150 unanswerable), so that cannot be
the whole story. For every false answer (unanswerable question that was answered) this
script finds the retrieved passage the answer/quote comes from and reports:
  * its rank (1-5) and whether it is the designated paragraph or another passage
  * FAR split by "designated paragraph retrieved?" (hit / miss) per config
If Dense/Hybrid false answers mostly quote *other* passages, the better hypothesis is that
semantic retrievers fill ranks 2-5 with on-topic distractors that BM25 does not.

Part 2 — audit sheet. Writes every false answer to false_answer_audit.csv with the
question, answer, quote and source-passage snippet, plus an empty `label` column for the
manual check (suggested labels: label_noise | answer_in_other_passage | distractor_entity |
negation_or_antonym | other).

    python scripts/11_false_answer_audit.py                 # all variants
    RAG_OUTPUT_SUBDIR=outputs_v3 python scripts/11_false_answer_audit.py

Writes <output dir>/false_answer_sources.json and <output dir>/false_answer_audit.csv.
"""
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from raglib import abstention, config, data, metrics


def overlap(a: str, b: str) -> float:
    ta = set(metrics.normalize_answer(a).split())
    return len(ta & set(metrics.normalize_answer(b).split())) / len(ta) if ta else 0.0


def main():
    corpus = {d.id: d.text for d in data.load_corpus(config.CORPUS_PATH)}
    q = {r["qid"]: r for r in data.read_jsonl(config.QUERIES_PATH)}
    retr = {(r["method"], r["qid"]): r for r in data.read_jsonl(config.RETRIEVAL_PATH)}
    gens = data.read_jsonl(config.GENERATIONS_PATH)
    needed = {i for r in retr.values() for i in r["retrieved_doc_ids"][:config.TOP_K]}
    missing = needed - corpus.keys()
    if len(missing) > 0.05 * len(needed):
        sys.exit(f"[11] {len(missing)}/{len(needed)} retrieved passages are not in "
                 f"{config.CORPUS_PATH} — this needs the full 20k corpus (run it on the VM).")

    src = defaultdict(Counter)               # config -> where the false answers come from
    strat = defaultdict(lambda: {"hit": [0, 0], "miss": [0, 0]})   # [false answers, n]
    rows = []
    for g in gens:
        qq = q[g["qid"]]
        if not qq["is_impossible"]:
            continue
        cfg = f"{g['method']}|{g['variant']}"
        r = retr[(g["method"], g["qid"])]
        ids = r["retrieved_doc_ids"][:config.TOP_K]
        hit = "hit" if qq["gold_doc_id"] in ids else "miss"
        answered = not abstention.is_abstention(g["answer"])
        strat[cfg][hit][1] += 1
        if not answered:
            continue
        strat[cfg][hit][0] += 1
        probe = g.get("evidence_quote") or g["answer"]
        scores = [overlap(probe, corpus.get(i, "")) for i in ids]
        k = max(range(len(ids)), key=lambda j: scores[j]) if ids else 0
        kind = "designated" if ids and ids[k] == qq["gold_doc_id"] else "other_passage"
        src[cfg][f"{kind}@rank{k + 1}"] += 1
        src[cfg][kind] += 1
        rows.append({"config": cfg, "qid": g["qid"], "question": qq["question"],
                     "answer": g["answer"], "evidence_quote": g.get("evidence_quote", ""),
                     "source": kind, "source_rank": k + 1,
                     "source_snippet": corpus.get(ids[k], "")[:400] if ids else "",
                     "designated_retrieved": hit == "hit", "label": ""})

    out = {}
    print(f"false answers by source passage ({config.OUTPUT_DIR.name}/)")
    print(f"{'config':<17}{'#FA':>5}{'designated':>12}{'other':>8}   "
          f"{'FAR | designated hit':>22}{'FAR | miss':>12}")
    for cfg in sorted(strat):
        s, c = strat[cfg], src[cfg]
        n_fa = s["hit"][0] + s["miss"][0]
        far_hit = s["hit"][0] / s["hit"][1] if s["hit"][1] else 0.0
        far_miss = s["miss"][0] / s["miss"][1] if s["miss"][1] else 0.0
        out[cfg] = {"n_false_answers": n_fa, "sources": dict(c),
                    "far_designated_hit": far_hit, "n_hit": s["hit"][1],
                    "far_designated_miss": far_miss, "n_miss": s["miss"][1]}
        print(f"{cfg:<17}{n_fa:>5}{c['designated']:>12}{c['other_passage']:>8}   "
              f"{far_hit:>14.3f} (n={s['hit'][1]}){far_miss:>8.3f} (n={s['miss'][1]})")
    with open(config.OUTPUT_DIR / "false_answer_sources.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    path = config.OUTPUT_DIR / "false_answer_audit.csv"
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["config"])
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["qid"], r["config"])))
    print(f"[11] wrote false_answer_sources.json and {path.name} ({len(rows)} false answers; "
          f"fill the `label` column for the manual audit)")


if __name__ == "__main__":
    main()

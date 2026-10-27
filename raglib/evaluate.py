"""Aggregate retrieval + generation + grounding into the final metrics.

Single source of truth used by both scripts/05_evaluate.py and the offline
smoke test, so they can never drift apart.

The headline analysis is `stratified_by_recall`: it splits each question by
whether the gold paragraph was retrieved (Recall@k = 1 vs 0) and reports the
False Answer Rate and grounding within each stratum — directly linking retrieval
quality to downstream hallucination/abstention.
"""
from collections import defaultdict
from typing import Dict, List

from . import abstention, config, generator, metrics


def evaluate_all(retrieval_rows: List[dict], generation_rows: List[dict],
                 grounding_rows: List[dict], queries: List[dict],
                 top_k: int = 5) -> dict:
    qmap = {q["qid"]: q for q in queries}
    rmap = {(r["method"], r["qid"]): r for r in retrieval_rows}
    gmap = {g["key"]: g for g in grounding_rows}

    # ── Retrieval metrics per method (variant-independent) ──
    retrieval_metrics = {}
    for m in sorted({r["method"] for r in retrieval_rows}):
        rows = [r for r in retrieval_rows if r["method"] == m]
        retrieval_metrics[m] = {
            "recall@k": metrics.mean([metrics.recall_at_k(r["retrieved_doc_ids"], [r["gold_doc_id"]], top_k) for r in rows]),
            "mrr@k": metrics.mean([metrics.mrr_at_k(r["retrieved_doc_ids"], [r["gold_doc_id"]], top_k) for r in rows]),
            "n": len(rows),
        }

    # ── Per (method, variant) reliability metrics ──
    groups = defaultdict(list)
    for g in generation_rows:
        groups[(g["method"], g["variant"])].append(g)

    by_mv = {}
    for (m, v), gens in sorted(groups.items()):
        records = []
        grounding_answered, f1_answerable, em_answerable = [], [], []
        # over ALL answerable questions, abstention scores 0 (review C2): comparing F1 only
        # on answered items rewards a system for answering just the easy questions.
        all_f1, all_em, all_contains = [], [], []
        strat = {1: {"unans": 0, "false_ans": 0, "grounding": []},
                 0: {"unans": 0, "false_ans": 0, "grounding": []}}

        for gen in gens:
            q = qmap[gen["qid"]]
            is_imp = q["is_impossible"]
            abstained = abstention.is_abstention(gen["answer"])
            records.append({"is_impossible": is_imp, "abstained": abstained})

            rrow = rmap.get((m, gen["qid"]))
            rec = metrics.recall_at_k(rrow["retrieved_doc_ids"], [rrow["gold_doc_id"]], top_k) if rrow else 0.0
            rec_bin = 1 if rec >= 1.0 else 0
            gr = gmap.get(gen["key"])

            if not abstained:
                if gr is not None:
                    grounding_answered.append(gr["grounding_score"])
                    strat[rec_bin]["grounding"].append(gr["grounding_score"])
                if not is_imp:
                    f1_answerable.append(metrics.answer_f1(gen["answer"], q["gold_answer"]))
                    em_answerable.append(metrics.exact_match(gen["answer"], q["gold_answer"]))
            if not is_imp:
                answered = not abstained
                all_f1.append(metrics.answer_f1(gen["answer"], q["gold_answer"]) if answered else 0.0)
                all_em.append(metrics.exact_match(gen["answer"], q["gold_answer"]) if answered else 0.0)
                all_contains.append(metrics.contains_gold(gen["answer"], q["gold_answer"]) if answered else 0.0)
            if is_imp:
                strat[rec_bin]["unans"] += 1
                if not abstained:
                    strat[rec_bin]["false_ans"] += 1

        counts = abstention.build_confusion(records)
        strat_out = {}
        for b in (1, 0):
            s = strat[b]
            strat_out[f"recall={b}"] = {
                "false_answer_rate": (s["false_ans"] / s["unans"]) if s["unans"] else 0.0,
                "n_unanswerable": s["unans"],
                "mean_grounding_answered": metrics.mean(s["grounding"]),
                "n_answered": len(s["grounding"]),
            }

        entry = {}
        if v == "cite":   # why cite abstained (review C1): designed vs format failure
            outc = defaultdict(int)
            for gen in gens:
                outc[generator.infer_cite_outcome(gen)] += 1
            entry["cite_outcomes"] = dict(outc)
        by_mv[f"{m}|{v}"] = {
            **entry,
            "confusion": counts,
            "rates": abstention.compute_rates(counts),
            "mean_grounding_answered": metrics.mean(grounding_answered),
            "f1_mode": config.F1_MODE,
            # conditional: only answered answerable questions (the v2 headline metric)
            "mean_token_f1_answerable": metrics.mean(f1_answerable),
            "mean_em_answerable": metrics.mean(em_answerable),
            # unconditional: all answerable questions, abstention = 0
            "f1_all_answerable": metrics.mean(all_f1),
            "em_all_answerable": metrics.mean(all_em),
            "contains_gold_all_answerable": metrics.mean(all_contains),
            "stratified_by_recall": strat_out,
        }

    return {"retrieval": retrieval_metrics, "by_method_variant": by_mv}

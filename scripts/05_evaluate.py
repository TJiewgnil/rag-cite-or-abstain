"""Step 05 — final evaluation (offline, free, re-runnable).

Joins retrieval + generations + grounding and reports:
  - Recall@5 / MRR@5 per retriever
  - 2x2 abstention matrix + False Answer Rate per (method, variant)
  - grounding score; Token F1 (auxiliary)
  - the headline stratified analysis: FAR & grounding split by Recall@5 = 1 vs 0

    python scripts/05_evaluate.py [--no-figures]

Writes outputs/summary.json (+ figures to outputs/figs/ if matplotlib present).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from raglib import config, data, evaluate


def _print_summary(summary: dict, top_k: int):
    print("\n==================== RETRIEVAL ====================")
    for m, v in summary["retrieval"].items():
        print(f"  {m:<7} Recall@{top_k}={v['recall@k']:.3f}  MRR@{top_k}={v['mrr@k']:.3f}  (n={v['n']})")

    print("\n============ RELIABILITY (method | variant) ============")
    for mv, r in summary["by_method_variant"].items():
        rates = r["rates"]
        print(f"\n  [{mv}]")
        print(f"    False Answer Rate = {rates['false_answer_rate']:.3f}   "
              f"Correct Abstention = {rates['correct_abstention_rate']:.3f}   "
              f"False Refusal = {rates['false_refusal_rate']:.3f}")
        print(f"    grounding(answered) = {r['mean_grounding_answered']:.3f}   "
              f"F1[{r['f1_mode']}] answered = {r['mean_token_f1_answerable']:.3f}   "
              f"all answerable (abstain=0): F1 = {r['f1_all_answerable']:.3f}  "
              f"EM = {r['em_all_answerable']:.3f}  contains-gold = {r['contains_gold_all_answerable']:.3f}")
        print(f"    confusion = {r['confusion']}")
        for stratum, s in r["stratified_by_recall"].items():
            print(f"      {stratum}: FAR={s['false_answer_rate']:.3f} "
                  f"(n_unans={s['n_unanswerable']})  "
                  f"grounding={s['mean_grounding_answered']:.3f} (n_ans={s['n_answered']})")


def _save_figures(summary: dict):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[05] matplotlib unavailable ({e}); skipping figures")
        return
    figs = config.OUTPUT_DIR / "figs"
    figs.mkdir(exist_ok=True)

    mvs = list(summary["by_method_variant"].keys())
    fars = [summary["by_method_variant"][mv]["rates"]["false_answer_rate"] for mv in mvs]
    plt.figure(figsize=(7, 4))
    plt.bar(mvs, fars, color="#c0392b")
    plt.ylabel("False Answer Rate"); plt.title("FAR by method | variant")
    plt.xticks(rotation=30, ha="right"); plt.tight_layout()
    plt.savefig(figs / "far_by_method_variant.png", dpi=120); plt.close()
    print(f"[05] figures -> {figs}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top_k", type=int, default=config.TOP_K)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    summary = evaluate.evaluate_all(
        retrieval_rows=data.read_jsonl(config.RETRIEVAL_PATH),
        generation_rows=data.read_jsonl(config.GENERATIONS_PATH),
        grounding_rows=data.read_jsonl(config.GROUNDING_PATH),
        queries=data.read_jsonl(config.QUERIES_PATH),
        top_k=args.top_k,
    )
    _print_summary(summary, args.top_k)

    out = config.OUTPUT_DIR / "summary.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[05] summary -> {out}")

    if not args.no_figures:
        _save_figures(summary)


if __name__ == "__main__":
    main()

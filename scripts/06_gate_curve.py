"""Step 06 — Grounding-gate threshold curve (offline, free, re-runnable).

A post-hoc reliability knob: override an answer to an abstention when its
claim-grounding score is <= a cut theta. We sweep theta over the observed
grounding scores and, for each (method, variant), recompute the abstention 2x2
and a single combined utility -- so you can read a risk-coverage curve and pick
an operating point *without re-running generation* (no Gemini cost, no GPU).

Combined utility (SQuAD-v2 style, tunable via CLI):
    answerable   & answered   ->  token_f1(answer, gold)
    answerable   & abstained  -> -fr_penalty     (false refusal;      default 0)
    unanswerable & abstained  -> +cab_reward      (correct abstention; default 1)
    unanswerable & answered   -> -fa_penalty      (false answer;       default 0)
Defaults reproduce the SQuAD-v2 F1 objective (no free weights). Raise
--fa_penalty to lean on reliability (punish hallucination harder).

    python scripts/06_gate_curve.py
    python scripts/06_gate_curve.py --fa_penalty 1.0     # reliability-leaning
    python scripts/06_gate_curve.py --target_far 0.10    # also report tau at FAR<=0.10

Reads outputs/{retrieval,generations,grounding}.jsonl + data/queries.jsonl.
Writes outputs/gate_curve.json (+ figures to outputs/figs/ if matplotlib present).
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from raglib import abstention, config, data, metrics


def grounding_signal(gr, signal: str, support_threshold: float) -> float:
    """Pull the gating signal out of a grounding row.

    signal='score' -> fraction of claims with max-entailment >= support_threshold.
        Recomputed from the stored per-claim `claim_max_entail` when present, so the
        support cutoff can be swept offline; falls back to the stored grounding_score.
    signal='mean'/'max' -> continuous mean / max of the per-claim max-entailment,
        a smoother signal than the near-binary supported-fraction.
    """
    if gr is None:
        return 0.0
    cme = gr.get("claim_max_entail")
    if signal == "mean":
        if cme:
            return sum(cme) / len(cme)
        return gr.get("grounding_mean", gr.get("grounding_score", 0.0))
    if signal == "max":
        if cme:
            return max(cme)
        return gr.get("grounding_max", gr.get("grounding_score", 0.0))
    if signal in ("qa_mean", "qa_max", "quote"):   # question-aware / quote-match (Track 3)
        key = {"qa_mean": "grounding_qa_mean", "qa_max": "grounding_qa_max",
               "quote": "quote_match"}[signal]
        return float(gr.get(key, 0.0))
    # signal == "score"
    if cme:
        return sum(1 for x in cme if x >= support_threshold) / len(cme)
    return gr.get("grounding_score", 0.0)


def make_utility(cab_reward: float, fr_penalty: float, fa_penalty: float):
    """Map a 2x2 outcome (+ token-F1 for answered-answerable) to a scalar reward."""
    def util(outcome: str, f1: float) -> float:
        if outcome == abstention.CORRECT_ANSWER:
            return f1
        if outcome == abstention.CORRECT_ABSTENTION:
            return cab_reward
        if outcome == abstention.FALSE_REFUSAL:
            return -fr_penalty
        return -fa_penalty  # FALSE_ANSWER
    return util


def curve_for_group(gens, qmap, gmap, util, sig) -> list:
    """Sweep the grounding cut theta for one (method, variant); one point per theta.

    Gate rule: an answer becomes an abstention when its grounding signal <= theta.
    theta = -1 gates nothing (baseline); theta = max signal gates everything.
    `sig(grounding_row) -> float` selects which grounding signal to gate on.
    """
    items = []
    for g in gens:
        q = qmap[g["qid"]]
        gr = gmap.get(g["key"])
        gs = sig(gr)
        base_abstain = abstention.is_abstention(g["answer"])
        f1 = metrics.answer_f1(g["answer"], q["gold_answer"]) if not q["is_impossible"] else 0.0
        items.append({"is_imp": q["is_impossible"], "base_abstain": base_abstain,
                      "gs": gs, "f1": f1})

    n = len(items)
    thetas = sorted({-1.0} | {it["gs"] for it in items})

    points = []
    for theta in thetas:
        c = {abstention.CORRECT_ANSWER: 0, abstention.FALSE_REFUSAL: 0,
             abstention.FALSE_ANSWER: 0, abstention.CORRECT_ABSTENTION: 0}
        u = 0.0
        for it in items:
            abstain = it["base_abstain"] or (it["gs"] <= theta)
            outcome = abstention.classify_outcome(it["is_imp"], abstain)
            c[outcome] += 1
            u += util(outcome, it["f1"])
        rates = abstention.compute_rates(c)
        answered = c[abstention.CORRECT_ANSWER] + c[abstention.FALSE_ANSWER]
        points.append({
            "theta": theta,
            "far": rates["false_answer_rate"],
            "frr": rates["false_refusal_rate"],
            "correct_answer_rate": rates["correct_answer_rate"],
            "correct_abstention_rate": rates["correct_abstention_rate"],
            "coverage": answered / n if n else 0.0,
            "utility": u / n if n else 0.0,
            "confusion": dict(c),
        })
    return points


def pick_operating_points(points: list, target_far: float | None) -> dict:
    """Baseline (no gate), best-utility, and (optionally) best-utility at FAR<=target."""
    baseline = min(points, key=lambda p: p["theta"])          # theta == -1
    best_utility = max(points, key=lambda p: p["utility"])
    ops = {"baseline": baseline, "best_utility": best_utility}
    if target_far is not None:
        feasible = [p for p in points if p["far"] <= target_far]
        ops["target_far"] = max(feasible, key=lambda p: p["utility"]) if feasible else None
        ops["target_far_value"] = target_far
    return ops


def _fmt(p: dict) -> str:
    return (f"theta<={p['theta']:+.3f}  util={p['utility']:.3f}  "
            f"FAR={p['far']:.3f}  FRR={p['frr']:.3f}  cov={p['coverage']:.3f}")


def _print_report(results: dict, target_far):
    print("\n==================== GROUNDING-GATE CURVE ====================")
    for mv, r in results.items():
        ops = r["operating_points"]
        b, bu = ops["baseline"], ops["best_utility"]
        print(f"\n  [{mv}]")
        print(f"    baseline (no gate) : {_fmt(b)}")
        print(f"    best utility       : {_fmt(bu)}   "
              f"(dUtil={bu['utility'] - b['utility']:+.3f}, dFAR={bu['far'] - b['far']:+.3f})")
        if target_far is not None:
            tf = ops.get("target_far")
            print(f"    best @ FAR<={target_far:.2f}   : {_fmt(tf) if tf else 'infeasible'}")


def _save_figures(results: dict):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[06] matplotlib unavailable ({e}); skipping figures")
        return
    figs = config.OUTPUT_DIR / "figs"
    figs.mkdir(exist_ok=True)

    # (1) Utility vs grounding cut theta.
    plt.figure(figsize=(8, 5))
    for mv, r in results.items():
        pts = [p for p in r["points"] if p["theta"] >= 0.0]
        plt.plot([p["theta"] for p in pts], [p["utility"] for p in pts],
                 marker="o", ms=3, label=mv)
        bu = r["operating_points"]["best_utility"]
        if bu["theta"] >= 0.0:
            plt.scatter([bu["theta"]], [bu["utility"]], s=70, zorder=5,
                        edgecolor="black", facecolor="none")
    plt.xlabel("grounding cut  theta  (gate if grounding <= theta)")
    plt.ylabel("mean combined utility")
    plt.title("Grounding-gate: utility vs threshold")
    plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(figs / "gate_curve_utility.png", dpi=120); plt.close()

    # (2) Risk-coverage: FAR vs answerable coverage (correct_answer_rate).
    plt.figure(figsize=(8, 5))
    for mv, r in results.items():
        pts = sorted(r["points"], key=lambda p: p["correct_answer_rate"])
        plt.plot([p["correct_answer_rate"] for p in pts], [p["far"] for p in pts],
                 marker=".", ms=4, label=mv)
        b = r["operating_points"]["baseline"]
        plt.scatter([b["correct_answer_rate"]], [b["far"]], s=50, marker="x", zorder=5)
    plt.xlabel("answerable coverage  (correct_answer_rate)")
    plt.ylabel("False Answer Rate")
    plt.title("Grounding-gate: risk-coverage  (x = baseline, no gate)")
    plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(figs / "gate_curve_risk_coverage.png", dpi=120); plt.close()
    print(f"[06] figures -> {figs}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cab_reward", type=float, default=1.0, help="reward for a correct abstention")
    ap.add_argument("--fr_penalty", type=float, default=0.0, help="penalty for a false refusal")
    ap.add_argument("--fa_penalty", type=float, default=0.0, help="penalty for a false answer")
    ap.add_argument("--target_far", type=float, default=None, help="also report best utility s.t. FAR<=this")
    ap.add_argument("--signal",
                    choices=["score", "mean", "max", "qa_mean", "qa_max", "quote"],
                    default="score",
                    help="gating signal: binary supported-fraction (score); continuous "
                         "mean/max per-claim entailment; question-aware qa_mean/qa_max; "
                         "or quote (evidence-quote match, cite variant)")
    ap.add_argument("--support_threshold", type=float, default=config.GROUNDING_THRESHOLD,
                    help="with --signal score, (re)binarize claims at this entailment prob "
                         "(uses stored claim_max_entail when present, so it sweeps offline)")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    qmap = {q["qid"]: q for q in data.read_jsonl(config.QUERIES_PATH)}
    gmap = {g["key"]: g for g in data.read_jsonl(config.GROUNDING_PATH)}

    groups = defaultdict(list)
    for g in data.read_jsonl(config.GENERATIONS_PATH):
        groups[(g["method"], g["variant"])].append(g)

    util = make_utility(args.cab_reward, args.fr_penalty, args.fa_penalty)
    sig = lambda gr: grounding_signal(gr, args.signal, args.support_threshold)
    print(f"[06] signal={args.signal}"
          + (f" (support_threshold={args.support_threshold})" if args.signal == "score" else "")
          + f"  weights(cab={args.cab_reward}, fr={args.fr_penalty}, fa={args.fa_penalty})")
    results = {}
    for (m, v), gens in sorted(groups.items()):
        points = curve_for_group(gens, qmap, gmap, util, sig)
        results[f"{m}|{v}"] = {
            "points": points,
            "operating_points": pick_operating_points(points, args.target_far),
        }

    _print_report(results, args.target_far)

    out = {
        "signal": args.signal,
        "support_threshold": args.support_threshold,
        "weights": {"cab_reward": args.cab_reward, "fr_penalty": args.fr_penalty,
                    "fa_penalty": args.fa_penalty},
        "target_far": args.target_far,
        "by_method_variant": results,
    }
    out_path = config.OUTPUT_DIR / "gate_curve.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\n[06] curve -> {out_path}")

    if not args.no_figures:
        _save_figures(results)


if __name__ == "__main__":
    main()

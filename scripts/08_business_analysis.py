"""Step 08 — business-oriented analysis: cascades + cost-weighting crossover.

Offline, free, re-runnable. Addresses "is ~50% answerable-coverage too low for a
real business?" with two rigorous pieces.

Business utility per question (this is NOT the SQuAD-v2 metric — see note):
    correct answer       +token_f1   (served the customer, scaled by quality)
    false answer         -beta       (gave wrong info)
    false refusal        -gamma      (refused an answerable question = a miss)
    correct abstention    0          (unanswerable; declining is neither win nor loss)
High beta / low gamma  = reliability regime (wrong answers are what hurt).
High gamma             = business regime (you must serve; missing a customer hurts).
NOTE: the SQuAD-v2 metric used earlier instead *rewards* correct abstention (+1),
which is exactly why it favoured the low-coverage `cite`; here declining earns 0.

ITEM 1 — Smart cascade. Plain cascade = `cite` when it answers else `explicit`.
Smart cascade also filters the explicit fallback through a question-aware grounding
gate (keep only if grounding_qa_mean >= tau). We test whether that gate buys a
genuinely better coverage/FAR tradeoff than the plain cite->cascade interpolation,
at matched answerable-coverage. If not, we keep the plain cascade.

ITEM 2 — Cost-weighting crossover. Sweep gamma (and beta) and show which strategy
is optimal where: a 1-D crossover (utility vs gamma, beta=1) and a 2-D optimal-
strategy map over (gamma, beta). Each strategy uses its best retriever per weighting.

    python scripts/08_business_analysis.py [--bootstrap 1000]

Reads outputs/{generations,grounding}.jsonl + data/queries.jsonl.
Writes outputs/business_analysis.json (+ figs/crossover.png, figs/optimal_regions.png).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from raglib import abstention, config, data, metrics

METHODS = ["BM25", "Dense", "Hybrid"]
FIXED = ["cite", "explicit", "naive", "cascade"]   # fixed policies for the crossover
METHODS_FOR_STRATEGY = METHODS                     # set by --retriever in main()


def load():
    q = {r["qid"]: r for r in data.read_jsonl(config.QUERIES_PATH)}
    ans, qa = {}, {}
    for g in data.read_jsonl(config.GENERATIONS_PATH):
        ans[(g["method"], g["variant"], g["qid"])] = g.get("answer", "")
    for gr in data.read_jsonl(config.GROUNDING_PATH):
        qa[(gr["method"], gr["variant"], gr["qid"])] = gr.get("grounding_qa_mean", 0.0)
    return q, ans, qa


def used_answer(strategy, method, qid, ans, qa, tau):
    def a(v):
        return ans.get((method, v, qid), "")
    if strategy in ("cite", "explicit", "naive"):
        return a(strategy)
    cite = a("cite")
    if not abstention.is_abstention(cite):
        return cite
    expl = a("explicit")
    if abstention.is_abstention(expl):
        return ""
    if strategy == "cascade":
        return expl
    return expl if qa.get((method, "explicit", qid), 0.0) >= tau else ""   # smart_cascade


def outcomes(strategy, method, qids, q, ans, qa, tau=-1.0):
    rows = []
    for qid in qids:
        txt = used_answer(strategy, method, qid, ans, qa, tau)
        imp = bool(q[qid]["is_impossible"])
        answered = bool(txt.strip()) and not abstention.is_abstention(txt)
        f1 = metrics.answer_f1(txt, q[qid].get("gold_answer", "")) if (answered and not imp) else 0.0
        rows.append((imp, answered, f1))
    return rows


def agg(rows):
    ca = fr = fa = cab = 0
    sf1 = 0.0
    for imp, answered, f1 in rows:
        if answered:
            if imp: fa += 1
            else: ca += 1; sf1 += f1
        else:
            if imp: cab += 1
            else: fr += 1
    n = len(rows)
    return {"ca": ca, "fr": fr, "fa": fa, "cab": cab, "sum_f1": sf1,
            "car": ca / (ca + fr) if (ca + fr) else 0.0,
            "far": fa / (fa + cab) if (fa + cab) else 0.0,
            "cov": (ca + fa) / n}


def util(rows, beta, gamma):
    u = 0.0
    for imp, answered, f1 in rows:
        if answered:
            u += (-beta) if imp else f1
        else:
            u += 0.0 if imp else (-gamma)
    return u / len(rows)


def boot_ci(rows, fn, B, seed=0):
    rng = np.random.default_rng(seed)
    n = len(rows)
    v = np.sort([fn([rows[i] for i in rng.integers(0, n, n)]) for _ in range(B)])
    return [round(float(v[int(0.025 * B)]), 3), round(float(v[int(0.975 * B)]), 3)]


def smart_curve(method, qids, q, ans, qa):
    """(car, far) over the smart-cascade gate threshold tau (ascending car)."""
    taus = [-1.0] + sorted({qa.get((method, "explicit", qid), 0.0) for qid in qids})
    pts = []
    for t in taus:
        st = agg(outcomes("smart_cascade", method, qids, q, ans, qa, t))
        pts.append((st["car"], st["far"], t))
    return sorted(pts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--retriever", default="best", choices=["best"] + METHODS,
                    help="'best' = best retriever per strategy chosen on the test data (v2, "
                         "optimistic); a fixed retriever avoids that selection (review C3). "
                         "Per-retriever crossover CIs: scripts/10_paired_tests.py")
    args = ap.parse_args()
    global METHODS_FOR_STRATEGY
    METHODS_FOR_STRATEGY = METHODS if args.retriever == "best" else [args.retriever]
    B = args.bootstrap
    q, ans, qa = load()
    qids = list(q)

    rows = {(s, m): outcomes(s, m, qids, q, ans, qa) for s in FIXED for m in METHODS}
    report = {"note": "business utility: CA=+f1, FA=-beta, FR=-gamma, CAbst=0", "item1": {}, "crossover": {}}

    # ---------------- ITEM 1: smart cascade, matched-coverage test ----------------
    print("==== ITEM 1: does the qa-gated 'smart cascade' beat the plain cite->explicit cascade? ====")
    print("(test at the midpoint answerable-coverage between cite and plain cascade; "
          "smart better iff its FAR is below the cite->cascade straight-line interpolation)")
    print(f"{'method':<8}{'mid car':>8}{'smart FAR':>11}{'linear FAR':>12}{'verdict':>12}")
    smart_wins = 0
    for m in METHODS:
        c = agg(rows[("cite", m)]); k = agg(rows[("cascade", m)])
        mid = (c["car"] + k["car"]) / 2
        curve = smart_curve(m, qids, q, ans, qa)
        # smart FAR at car closest to mid
        sp = min(curve, key=lambda p: abs(p[0] - mid))
        # linear interpolation of FAR between cite and cascade at car=sp[0]
        if k["car"] > c["car"]:
            lin = c["far"] + (sp[0] - c["car"]) / (k["car"] - c["car"]) * (k["far"] - c["far"])
        else:
            lin = c["far"]
        win = sp[1] < lin - 1e-6
        smart_wins += int(win)
        report["item1"][m] = {"cite": {"car": c["car"], "far": c["far"]},
                              "cascade": {"car": k["car"], "far": k["far"]},
                              "smart_at_mid": {"car": round(sp[0], 3), "far": round(sp[1], 3),
                                               "linear_far": round(lin, 3), "better": win}}
        print(f"{m:<8}{sp[0]:>8.2f}{sp[1]:>11.3f}{lin:>12.3f}{('better' if win else 'no gain'):>12}")
    use_smart = smart_wins >= 2
    report["item1_decision"] = {"smart_wins": smart_wins, "keep_smart": use_smart}
    print(f"[item1] smart cascade better on {smart_wins}/3 retrievers -> "
          f"{'KEEP smart as the coverage knob' if use_smart else 'KEEP plain cascade (no real gain)'}\n")

    # ---------------- ITEM 2: crossover ----------------
    def su(s, beta, gamma):                      # best retriever for strategy s at (beta,gamma)
        return max(util(rows[(s, m)], beta, gamma) for m in METHODS_FOR_STRATEGY)

    print("==== ITEM 2: strategy utility vs gamma (miss cost), beta=1 ====")
    print(f"{'gamma':>6}" + "".join(f"{s:>11}" for s in FIXED) + f"{'  best':>12}")
    for g in [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]:
        us = {s: su(s, 1.0, g) for s in FIXED}
        best = max(us, key=us.get)
        print(f"{g:>6.2f}" + "".join(f"{us[s]:>11.3f}" for s in FIXED) + f"{best:>12}")
        report["crossover"][f"gamma={g}"] = {"util": {s: round(us[s], 3) for s in FIXED}, "best": best}

    # bootstrap CI on the headline per-strategy utility at a balanced point (beta=1,gamma=1), best retriever
    report["ci_beta1_gamma1"] = {}
    for s in FIXED:
        m_best = max(METHODS_FOR_STRATEGY, key=lambda m: util(rows[(s, m)], 1.0, 1.0))
        report["ci_beta1_gamma1"][s] = {
            "retriever": m_best, "util": round(util(rows[(s, m_best)], 1.0, 1.0), 3),
            "util_ci": boot_ci(rows[(s, m_best)], lambda r: util(r, 1.0, 1.0), B)}

    _figs(su, rows)

    outp = config.OUTPUT_DIR / "business_analysis.json"
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[08] wrote {outp}")


def _figs(su, rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.colors import ListedColormap
    except Exception as e:
        print(f"[08] matplotlib unavailable ({e}); skipping figures")
        return
    figs = config.OUTPUT_DIR / "figs"; figs.mkdir(exist_ok=True)
    col = {"cite": "#1b7837", "explicit": "#2166ac", "naive": "#b2182b", "cascade": "#d95f02"}

    # Fig A: utility vs gamma (beta=1)
    gs = np.linspace(0, 3, 61)
    plt.figure(figsize=(7.6, 4.6))
    for s in FIXED:
        plt.plot(gs, [su(s, 1.0, g) for g in gs], lw=2, color=col[s], label=s)
    plt.plot(gs, [max(su(s, 1.0, g) for s in FIXED) for g in gs], "k--", lw=1, alpha=.6, label="optimal")
    plt.axhline(0, color="gray", lw=.5)
    plt.xlabel("gamma  =  cost of refusing an answerable question  (0 = reliability,  high = must-serve business)")
    plt.ylabel("business utility (best retriever)")
    plt.title("Which strategy wins as 'missing a customer' gets costlier (beta=1)")
    plt.legend(fontsize=8); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(figs / "crossover.png", dpi=120); plt.close()

    # Fig B: optimal strategy over (gamma, beta)
    gg = np.linspace(0, 3, 61); bb = np.linspace(0, 3, 61)
    grid = np.zeros((len(bb), len(gg)), int)
    for bi, beta in enumerate(bb):
        for gi, g in enumerate(gg):
            grid[bi, gi] = int(np.argmax([su(s, beta, g) for s in FIXED]))
    uniq = sorted(set(grid.flatten()))
    cmap = ListedColormap([col[FIXED[i]] for i in range(len(FIXED))])
    plt.figure(figsize=(7.6, 5))
    plt.imshow(grid, origin="lower", aspect="auto", cmap=cmap, vmin=0, vmax=len(FIXED) - 1,
               extent=[gg[0], gg[-1], bb[0], bb[-1]])
    plt.legend(handles=[mpatches.Patch(color=col[FIXED[i]], label=FIXED[i]) for i in uniq],
               fontsize=8, loc="upper right")
    plt.xlabel("gamma  =  cost of missing an answerable question")
    plt.ylabel("beta  =  cost of a wrong answer")
    plt.title("Optimal strategy by business priorities (color = best strategy)")
    plt.tight_layout(); plt.savefig(figs / "optimal_regions.png", dpi=120); plt.close()
    print(f"[08] figures -> {figs}")


if __name__ == "__main__":
    main()

"""Step 07 — multi-signal learned abstention gate (Track 4) + AURC & bootstrap (Track 5).

Offline, free, re-runnable. For each (method, variant):
  - Builds per-answer features: grounding (answer-only + question-aware), evidence-
    quote match, retrieval scores (top / gap / mean), answer shape, has_answer.
  - Learns a small 'should-abstain' logistic (5-fold CV, no leakage) that combines
    them, and gates answers by its confidence.
  - Compares no-gate vs best single signal (question-aware grounding) vs the learned
    multi-signal gate, by utility / FAR / coverage and AURC (area under the
    risk-coverage curve; lower = better).
  - Reports bootstrap 95% CIs so the n=300 differences can be read with uncertainty.

    python scripts/07_multisignal_gate.py
    python scripts/07_multisignal_gate.py --fa_penalty 1.0 --bootstrap 1000

Reads outputs/{generations,grounding,retrieval}.jsonl + data/queries.jsonl.
Writes outputs/multisignal_gate.json.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from raglib import abstention, config, data, metrics

# np.trapz was removed in NumPy 2.x; np.trapezoid does not exist before 2.0.
_trapezoid = getattr(np, "trapezoid", None) or getattr(np, "trapz")

FEATURES = ["grounding_mean", "grounding_max", "grounding_qa_mean", "grounding_qa_max",
            "quote_match", "n_claims", "has_answer", "quote_match_gen",
            "ans_len", "ret_top", "ret_gap", "ret_mean"]


def load_groups():
    gmap = {g["key"]: g for g in data.read_jsonl(config.GROUNDING_PATH)}
    rmap = {(r["method"], r["qid"]): r for r in data.read_jsonl(config.RETRIEVAL_PATH)}
    qmap = {q["qid"]: q for q in data.read_jsonl(config.QUERIES_PATH)}
    groups = defaultdict(list)
    for g in data.read_jsonl(config.GENERATIONS_PATH):
        q = qmap[g["qid"]]
        gr = gmap.get(g["key"], {})
        scores = (rmap.get((g["method"], g["qid"]), {}) or {}).get("scores") or [0.0]
        ans = g.get("answer", "")
        base_ans = not abstention.is_abstention(ans)
        is_imp = bool(q["is_impossible"])
        feat = {
            "grounding_mean": gr.get("grounding_mean", 0.0),
            "grounding_max": gr.get("grounding_max", 0.0),
            "grounding_qa_mean": gr.get("grounding_qa_mean", 0.0),
            "grounding_qa_max": gr.get("grounding_qa_max", 0.0),
            "quote_match": gr.get("quote_match", 0.0),
            "n_claims": gr.get("n_claims", 0),
            "has_answer": 1.0 if g.get("has_answer", base_ans) else 0.0,
            "quote_match_gen": g.get("quote_match_gen", 0.0),
            "ans_len": float(len(ans.split())),
            "ret_top": scores[0],
            "ret_gap": scores[0] - scores[1] if len(scores) > 1 else scores[0],
            "ret_mean": sum(scores) / len(scores),
        }
        groups[f"{g['method']}|{g['variant']}"].append({
            "is_imp": is_imp, "base_ans": base_ans,
            "f1": metrics.answer_f1(ans, q.get("gold_answer", "")) if not is_imp else 0.0,
            "feat": feat,
        })
    return groups


def confusion_util(items, answered, fa, fr=0.0, cab=1.0):
    c = {"ca": 0, "fr": 0, "fa": 0, "cab": 0}
    u = 0.0
    for it, a in zip(items, answered):
        if a:
            if it["is_imp"]:
                c["fa"] += 1; u -= fa
            else:
                c["ca"] += 1; u += it["f1"]
        else:
            if it["is_imp"]:
                c["cab"] += 1; u += cab
            else:
                c["fr"] += 1; u -= fr
    n = len(items)
    n_un = c["fa"] + c["cab"]
    return u / n, (c["fa"] / n_un if n_un else 0.0), (c["ca"] + c["fa"]) / n


def best_gate(items, score, fa):
    """Sweep an accept threshold on `score` (higher = safer); gate acts on base-answered."""
    base = np.array([it["base_ans"] for it in items])
    sc = np.asarray(score, float)
    cands = [-np.inf] + sorted(set(sc[base].tolist())) if base.any() else [-np.inf]
    best = None
    for t in cands:
        ans = base & (sc >= t)
        u, far, cov = confusion_util(items, ans, fa)
        if best is None or u > best["util"]:
            best = {"t": float(t), "util": u, "far": far, "cov": cov,
                    "answered": ans.copy()}
    return best


def aurc(items, score, legacy=False):
    """Area under the risk-coverage curve of the gate over the base-answered pool
    (lower = the score ranks good answers first).

    Coverage runs from 0 to 1 over that pool. v2 divided by ALL items instead
    (legacy=True), so a config that answers fewer questions integrated over a shorter
    range and got a smaller AURC mechanically (review C3).
    """
    base = np.array([it["base_ans"] for it in items])
    loss = np.array([1.0 if it["is_imp"] else (1.0 - it["f1"]) for it in items])
    idx = np.where(base)[0]
    if idx.size == 0:
        return 0.0
    denom = len(items) if legacy else idx.size
    order = idx[np.argsort(-np.asarray(score, float)[idx])]   # most-confident first
    cov, risk, cum = [0.0], [0.0], 0.0
    for k, i in enumerate(order, 1):
        cum += loss[i]
        cov.append(k / denom); risk.append(cum / k)
    return float(_trapezoid(risk, cov))


def nested_cv_gate(items, fa, score_fn, folds=5, seed=0):
    """Out-of-sample gate utility (review C3).

    best_gate() picks the accept threshold on the same items it is scored on, which is
    optimistic. Here the threshold (and, for the learned gate, the logistic model) is
    fit on 4 folds and applied to the held-out fold; the held-out decisions are pooled.
    score_fn(train_items, test_items) -> (train_scores, test_scores).
    """
    n = len(items)
    y = np.array([1 if it["is_imp"] else 0 for it in items])
    answered = np.zeros(n, dtype=bool)
    cv = StratifiedKFold(folds, shuffle=True, random_state=seed)
    for tr, te in cv.split(np.zeros(n), y):
        tr_items, te_items = [items[i] for i in tr], [items[i] for i in te]
        s_tr, s_te = score_fn(tr_items, te_items)
        t = best_gate(tr_items, s_tr, fa)["t"]
        base_te = np.array([it["base_ans"] for it in te_items])
        answered[te] = base_te & (np.asarray(s_te, float) >= t)
    u, far, cov = confusion_util(items, answered, fa)
    return {"util": u, "far": far, "cov": cov, "answered": answered}


def qa_score_fn(tr_items, te_items):
    f = lambda its: [it["feat"]["grounding_qa_mean"] for it in its]
    return f(tr_items), f(te_items)


def multi_score_fn(tr_items, te_items):
    """Logistic fit on the training folds' answered items; P(answerable) as the score."""
    def X(its):
        return np.array([[it["feat"][f] for f in FEATURES] for it in its])
    base = np.array([it["base_ans"] for it in tr_items])
    y = np.array([1 if it["is_imp"] else 0 for it in tr_items])[base]
    if y.sum() < 2 or (len(y) - y.sum()) < 2:          # too few of a class -> no gate
        return [1.0] * len(tr_items), [1.0] * len(te_items)
    pipe = make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=2000, class_weight="balanced"))
    pipe.fit(X(tr_items)[base], y)
    return (1.0 - pipe.predict_proba(X(tr_items))[:, 1]).tolist(), \
           (1.0 - pipe.predict_proba(X(te_items))[:, 1]).tolist()


def multisignal_score(items):
    """Out-of-fold P(answerable) from a logistic over all features; None if too few of a class."""
    base = np.array([it["base_ans"] for it in items])
    X = np.array([[it["feat"][f] for f in FEATURES] for it in items])[base]
    y = np.array([1 if it["is_imp"] else 0 for it in items])[base]
    if y.sum() < 5 or (len(y) - y.sum()) < 5:
        return None
    pipe = make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=2000, class_weight="balanced"))
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    p = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]  # P(impossible)
    score = np.full(len(items), -1.0)
    score[np.where(base)[0]] = 1.0 - p
    return score


def bootstrap_ci(items, answered, fa, B, seed=0):
    rng = np.random.default_rng(seed)
    n = len(items)
    us, fars = [], []
    for _ in range(B):
        bs = rng.integers(0, n, n)
        sub = [items[i] for i in bs]
        a = [answered[i] for i in bs]
        u, far, _ = confusion_util(sub, a, fa)
        us.append(u); fars.append(far)
    def ci(a):
        a = np.sort(a)
        return [round(float(a[int(0.025 * len(a))]), 3), round(float(a[int(0.975 * len(a))]), 3)]
    return ci(us), ci(fars)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fa_penalty", type=float, default=1.0, help="false-answer penalty in utility")
    ap.add_argument("--bootstrap", type=int, default=1000, help="bootstrap resamples for CIs (0=off)")
    args = ap.parse_args()
    fa = args.fa_penalty

    groups = load_groups()
    out = {}

    print(f"\n==== Per-config utility (fa_penalty={fa}, bootstrap={args.bootstrap}) ====")
    print(f"{'config':<17}{'util_fa0':>9}{'util_fa':>9}{'  util_fa CI':>16}"
          f"{'FAR':>7}{'  FAR CI':>14}{'cov':>7}{'tokF1':>7}")
    for mv, items in sorted(groups.items()):
        base = [it["base_ans"] for it in items]
        u0, _, _ = confusion_util(items, base, 0.0)
        u1, far, cov = confusion_util(items, base, fa)
        f1s = [it["f1"] for it in items if it["base_ans"] and not it["is_imp"]]
        tok = sum(f1s) / len(f1s) if f1s else 0.0
        uci, fci = (bootstrap_ci(items, base, fa, args.bootstrap) if args.bootstrap else ([None]*2, [None]*2))
        out[mv] = {"util_fa0": u0, "util_fa": u1, "util_fa_ci": uci,
                   "far": far, "far_ci": fci, "cov": cov, "token_f1": tok}
        print(f"{mv:<17}{u0:>9.3f}{u1:>9.3f}{str(uci):>16}{far:>7.3f}{str(fci):>14}{cov:>7.3f}{tok:>7.3f}")

    print(f"\n==== Gate comparison (best utility @ fa_penalty={fa}; AURC lower=better) ====")
    print("(qa-gate / multi-gate = threshold picked on the same data, an optimistic upper bound;"
          " *-cv = nested 5-fold CV, out-of-sample)")
    print(f"{'config':<17}{'no-gate':>9}{'qa-gate':>9}{'multi-gate':>11}"
          f"{'AURC_qa':>9}{'AURC_multi':>11}{'qa-cv':>10}{'multi-cv':>12}")
    for mv, items in sorted(groups.items()):
        qa_score = [it["feat"]["grounding_qa_mean"] for it in items]
        qa = best_gate(items, qa_score, fa)
        base = [it["base_ans"] for it in items]
        u1, _, _ = confusion_util(items, base, fa)
        ms = multisignal_score(items)
        if ms is not None:
            mg = best_gate(items, ms, fa)
            mg_u, aurc_m = mg["util"], aurc(items, ms)
            mg_ci, _ = (bootstrap_ci(items, mg["answered"], fa, args.bootstrap) if args.bootstrap else ([None]*2, None))
        else:
            mg_u, aurc_m, mg_ci = float("nan"), float("nan"), [None, None]
        aurc_q = aurc(items, qa_score)
        qa_cv = nested_cv_gate(items, fa, qa_score_fn)
        mg_cv = nested_cv_gate(items, fa, multi_score_fn)
        out[mv].update({"gate_qa_util": qa["util"], "gate_qa_far": qa["far"], "gate_qa_cov": qa["cov"],
                        "gate_multi_util": mg_u, "gate_multi_ci": mg_ci,
                        "aurc_qa": aurc_q, "aurc_multi": aurc_m,
                        "aurc_qa_legacy": aurc(items, qa_score, legacy=True),
                        "aurc_multi_legacy": aurc(items, ms, legacy=True) if ms is not None else float("nan"),
                        # out-of-sample (nested CV) gate results — use these in the paper
                        "gate_qa_cv_util": qa_cv["util"], "gate_qa_cv_far": qa_cv["far"],
                        "gate_multi_cv_util": mg_cv["util"], "gate_multi_cv_far": mg_cv["far"]})
        print(f"{mv:<17}{u1:>9.3f}{qa['util']:>9.3f}{mg_u:>11.3f}{aurc_q:>9.3f}{aurc_m:>11.3f}"
              f"{qa_cv['util']:>10.3f}{mg_cv['util']:>12.3f}")

    best = max(out.items(), key=lambda kv: max(
        kv[1]["util_fa"], kv[1].get("gate_qa_util", 0), kv[1].get("gate_multi_util", 0) or 0))
    print(f"\n[07] best config by utility(fa={fa}): {best[0]}  "
          f"(no-gate {best[1]['util_fa']:.3f}, qa-gate {best[1]['gate_qa_util']:.3f}, "
          f"multi-gate {best[1]['gate_multi_util']:.3f})")

    outp = config.OUTPUT_DIR / "multisignal_gate.json"
    with open(outp, "w", encoding="utf-8") as f:
        json.dump({"fa_penalty": fa, "features": FEATURES, "by_method_variant": out}, f, indent=2)
    print(f"[07] wrote {outp}")


if __name__ == "__main__":
    main()

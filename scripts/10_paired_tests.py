"""Step 10 — paired significance tests + crossover CI (review C3). Offline, free.

All nine configurations answer the SAME 300 questions, so differences between them
must be tested with paired methods, not by checking whether two independent CIs
overlap:
  * exact McNemar test on false answers (the 150 unanswerable questions)
  * paired bootstrap (one resample of question ids shared by both configs) for
    dFAR, d(answerable coverage), dF1 over all answerable (abstain = 0) and dU_rel(lambda=1)
  * the business crossover gamma* at which the cite->explicit cascade overtakes cite,
    per retriever (no picking the best retriever on test data), with a bootstrap CI

    python scripts/10_paired_tests.py [--bootstrap 2000]
    RAG_OUTPUT_SUBDIR=outputs_v3 python scripts/10_paired_tests.py

Writes <output dir>/paired_tests.json.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from raglib import abstention, config, data, metrics

METHODS = ["BM25", "Dense", "Hybrid"]
VARIANTS = ["naive", "explicit", "cite"]


def load():
    q = {r["qid"]: r for r in data.read_jsonl(config.QUERIES_PATH)}
    qids = sorted(q)
    ans = {(g["method"], g["variant"], g["qid"]): g["answer"]
           for g in data.read_jsonl(config.GENERATIONS_PATH)}
    imp = np.array([bool(q[i]["is_impossible"]) for i in qids])
    gold = [q[i].get("gold_answer", "") for i in qids]

    def vectors(texts):
        answered = np.array([bool(t.strip()) and not abstention.is_abstention(t) for t in texts])
        f1 = np.array([metrics.answer_f1(t, g) if (a and not im) else 0.0
                       for t, g, a, im in zip(texts, gold, answered, imp)])
        return answered, f1

    cfg = {}
    for m in METHODS:
        for v in VARIANTS:
            cfg[f"{m}|{v}"] = vectors([ans.get((m, v, i), "") for i in qids])
        casc = []
        for i in qids:                                   # cite, else explicit fallback
            c = ans.get((m, "cite", i), "")
            casc.append(c if not abstention.is_abstention(c) else
                        (ans.get((m, "explicit", i), "") if not abstention.is_abstention(
                            ans.get((m, "explicit", i), "")) else ""))
        cfg[f"{m}|cascade"] = vectors(casc)
    return imp, cfg


def stats(imp, answered, f1, idx):
    im, a, f = imp[idx], answered[idx], f1[idx]
    n_un, n_an = im.sum(), (~im).sum()
    fa = (a & im).sum()
    ca = (a & ~im).sum()
    return {"far": fa / n_un if n_un else 0.0,
            "coverage": ca / n_an if n_an else 0.0,
            "f1_all": f[~im].sum() / n_an if n_an else 0.0,
            "u_rel": (f[a & ~im].sum() + (~a & im).sum() - fa) / len(idx)}


def mcnemar_exact(x, y):
    """x, y: boolean arrays (event per item). Two-sided exact binomial p-value."""
    b, c = int((x & ~y).sum()), int((~x & y).sum())
    n = b + c
    if n == 0:
        return b, c, 1.0
    k = min(b, c)
    p = 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return b, c, min(1.0, p)


def biz_parts(imp, answered, f1, idx, beta=1.0):
    """U_biz(gamma) * N = A - gamma * FR, with A = sum F1 on CA - beta * #FA."""
    im, a, f = imp[idx], answered[idx], f1[idx]
    return f[a & ~im].sum() - beta * (a & im).sum(), (~a & ~im).sum()


def crossover(imp, cite, casc, idx):
    """gamma* = largest miss cost at which cite still beats the cascade (0 = never)."""
    a1, fr1 = biz_parts(imp, *cite, idx)
    a2, fr2 = biz_parts(imp, *casc, idx)
    if a1 <= a2:
        return 0.0                        # cascade at least as good even when misses are free
    if fr1 <= fr2:
        return float("inf")               # cite better at every gamma (not expected)
    return (a1 - a2) / (fr1 - fr2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    imp, cfg = load()
    n = len(imp)
    rng = np.random.default_rng(args.seed)
    boots = [rng.integers(0, n, n) for _ in range(args.bootstrap)]
    full = np.arange(n)
    ci = lambda a: [round(float(np.percentile(a, 2.5)), 3), round(float(np.percentile(a, 97.5)), 3)]

    pairs = []
    for m in METHODS:
        pairs += [(f"{m}|cite", f"{m}|explicit"), (f"{m}|cite", f"{m}|naive"),
                  (f"{m}|explicit", f"{m}|naive")]
    pairs += [("BM25|cite", "Dense|cite"), ("BM25|cite", "Hybrid|cite"), ("Dense|cite", "Hybrid|cite")]
    pairs = [(A, B) for A, B in pairs if A in cfg and B in cfg]

    out = {"f1_mode": config.F1_MODE, "bootstrap": args.bootstrap, "pairs": {}, "crossover": {}}
    print(f"paired tests ({config.OUTPUT_DIR.name}/, F1={config.F1_MODE}, B={args.bootstrap})")
    print(f"{'A vs B':<28}{'FA only A/B':>12}{'McNemar p':>11}"
          f"{'dFAR [95% CI]':>24}{'dCov [95% CI]':>24}{'dF1all [95% CI]':>24}{'dU_rel [95% CI]':>24}")
    for A, B in pairs:
        (aA, fA), (aB, fB) = cfg[A], cfg[B]
        b, c, p = mcnemar_exact(aA & imp, aB & imp)
        d = {k: [] for k in ("far", "coverage", "f1_all", "u_rel")}
        for idx in boots:
            sA, sB = stats(imp, aA, fA, idx), stats(imp, aB, fB, idx)
            for k in d:
                d[k].append(sA[k] - sB[k])
        sA, sB = stats(imp, aA, fA, full), stats(imp, aB, fB, full)
        row = {"fa_only_A": b, "fa_only_B": c, "mcnemar_p": p}
        for k in d:
            row[f"d_{k}"] = round(float(sA[k] - sB[k]), 3)
            row[f"d_{k}_ci"] = ci(d[k])
        out["pairs"][f"{A} vs {B}"] = row
        fmt = lambda k: f"{row['d_' + k]:+.3f} {row['d_' + k + '_ci']}"
        print(f"{A + ' vs ' + B:<28}{f'{b}/{c}':>12}{p:>11.4f}"
              f"{fmt('far'):>24}{fmt('coverage'):>24}{fmt('f1_all'):>24}{fmt('u_rel'):>24}")

    print("\nbusiness crossover gamma* (cite beats cascade only for gamma < gamma*; beta=1)")
    for m in METHODS:
        cite, casc = cfg[f"{m}|cite"], cfg[f"{m}|cascade"]
        g = crossover(imp, cite, casc, full)
        gb = np.array([crossover(imp, cite, casc, idx) for idx in boots])
        out["crossover"][m] = {"gamma_star": round(g, 3), "ci": ci(gb),
                               "share_cascade_always_better": round(float((gb == 0).mean()), 3)}
        print(f"  {m:<7} gamma* = {g:.3f}  95% CI {ci(gb)}  "
              f"(cascade >= cite at every gamma in {100 * (gb == 0).mean():.0f}% of resamples)")

    path = config.OUTPUT_DIR / "paired_tests.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[10] wrote {path}")


if __name__ == "__main__":
    main()

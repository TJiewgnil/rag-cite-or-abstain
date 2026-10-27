"""Step 12 — independent rigor checks on a results folder (read-only, offline, free).

Recomputes the main table from generations.jsonl (and checks it against summary.json),
Wilson CIs, exact McNemar / sign tests with Holm correction over 12 comparisons,
crossover gamma* and lambda* with bootstrap CIs, and generation determinism (30 diagnostic
keys generated twice; needs outputs_v3_diag/, skipped otherwise).

    python scripts/12_rigor_checks.py                             # checks the shipped outputs_v3/
    RAG_OUTPUT_SUBDIR=outputs python scripts/12_rigor_checks.py   # checks your own re-run
"""
import json, math, os, sys, statistics as st
from collections import Counter
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.environ.setdefault("RAG_OUTPUT_SUBDIR", "outputs_v3")
from raglib import abstention, metrics, generator

R = lambda p: [json.loads(l) for l in open(p, encoding="utf-8")]
Q = {x["qid"]: x for x in R("data/queries.jsonl")}
qids = sorted(Q)
G = {(g["method"], g["variant"], g["qid"]): g for g in R(f"{OUT}/generations.jsonl")}
S = json.load(open(f"{OUT}/summary.json"))
imp = np.array([Q[i]["is_impossible"] for i in qids])
gold = [Q[i]["gold_answer"] for i in qids]
M, V = ["BM25", "Dense", "Hybrid"], ["naive", "explicit", "cite"]


def vec(texts):
    a = np.array([bool(t.strip()) and not abstention.is_abstention(t) for t in texts])
    def per(fn):
        return np.array([fn(t, g) if (x and not im) else 0.0 for t, g, x, im in zip(texts, gold, a, imp)])
    return dict(a=a, f1=per(metrics.squad_f1), em=per(metrics.exact_match),
                cg=per(metrics.contains_gold), texts=texts)


C = {}
for m in M:
    for v in V:
        C[(m, v)] = vec([G[(m, v, i)]["answer"] for i in qids])
    casc = []
    for i in qids:
        c, e = G[(m, "cite", i)]["answer"], G[(m, "explicit", i)]["answer"]
        casc.append(c if not abstention.is_abstention(c) else
                    (e if not abstention.is_abstention(e) else "I don't know."))
    C[(m, "cascade")] = vec(casc)


def stats(d, idx=None):
    idx = np.arange(len(imp)) if idx is None else idx
    a, im, f1 = d["a"][idx], imp[idx], d["f1"][idx]
    nu, na = im.sum(), (~im).sum()
    fa, ca, cab = (a & im).sum(), (a & ~im).sum(), (~a & im).sum()
    return dict(FAR=fa / nu, cov=ca / na, F1all=f1[~im].sum() / na, F1ans=f1[a & ~im].sum() / max(ca, 1),
                EM=d["em"][idx][~im].sum() / na, CG=d["cg"][idx][~im].sum() / na,
                Urel=(f1[a & ~im].sum() + cab - fa) / len(idx), FA=int(fa), CA=int(ca))


print("== 1. Main table (independent recomputation) ==")
print(f"{'config':<17}{'FA':>4}{'FAR':>7}{'cov':>7}{'F1all':>7}{'EMall':>7}{'CGall':>7}{'F1ans':>7}{'Urel':>7}  medLen")
mism = 0
for m in M:
    for v in V + ["cascade"]:
        s = stats(C[(m, v)])
        ln = st.median(len(t.split()) for t, x in zip(C[(m, v)]["texts"], C[(m, v)]["a"]) if x)
        print(f"{m + '|' + v:<17}{s['FA']:>4}{s['FAR']:>7.3f}{s['cov']:>7.3f}{s['F1all']:>7.3f}{s['EM']:>7.3f}"
              f"{s['CG']:>7.3f}{s['F1ans']:>7.3f}{s['Urel']:>7.3f}  {ln}")
        if v != "cascade":
            r = S["by_method_variant"][f"{m}|{v}"]
            if abs(r["rates"]["false_answer_rate"] - s["FAR"]) > 1e-9 or abs(r["f1_all_answerable"] - s["F1all"]) > 1e-9:
                mism += 1
print("mismatches vs summary.json:", mism)


def wilson(k, n, z=1.96):
    p = k / n
    c = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    d = 1 + z * z / n
    return (p + z * z / (2 * n) - c) / d, (p + z * z / (2 * n) + c) / d


print("\n== 2. Wilson 95% CI of FAR (n=150 unanswerable) ==")
for m in M:
    parts_ = []
    for v in V:
        k = stats(C[(m, v)])["FA"]
        lo, hi = wilson(k, 150)
        parts_.append(f"{v}: {k}/150 [{lo:.3f},{hi:.3f}]")
    print(f"{m:<7}", "  ".join(parts_))


def binom_p(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def holm(ps):
    o = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        adj[i] = run
    return adj


print("\n== 3. Paired tests: exact McNemar on FA + exact sign test on per-question F1 (answerable); Holm over 12 ==")
pairs = ([((m, "cite"), (m, "explicit")) for m in M] + [((m, "cite"), (m, "naive")) for m in M] +
         [((m, "explicit"), (m, "naive")) for m in M] +
         [(("BM25", "cite"), ("Dense", "cite")), (("BM25", "cite"), ("Hybrid", "cite")),
          (("Dense", "cite"), ("Hybrid", "cite"))])
rows = []
for A, B in pairs:
    x, y = C[A]["a"] & imp, C[B]["a"] & imp
    b, c = int((x & ~y).sum()), int((~x & y).sum())
    fA, fB = C[A]["f1"][~imp], C[B]["f1"][~imp]
    w, l = int((fA > fB).sum()), int((fA < fB).sum())
    rows.append([A, B, b, c, binom_p(b, c), w, l, binom_p(w, l)])
h1, h2 = holm([r[4] for r in rows]), holm([r[7] for r in rows])
print(f"{'A vs B':<34}{'FA A-only/B-only':>17}{'p':>8}{'p_Holm':>8}   {'F1 A>B/A<B':>11}{'p':>9}{'p_Holm':>9}")
for r, a1, a2 in zip(rows, h1, h2):
    name = "|".join(r[0]) + " vs " + "|".join(r[1])
    print(f"{name:<34}{str(r[2]) + '/' + str(r[3]):>17}{r[4]:>8.4f}{a1:>8.4f}   "
          f"{str(r[5]) + '/' + str(r[6]):>11}{r[7]:>9.2g}{a2:>9.2g}")


def parts(d, idx):
    a, im, f = d["a"][idx], imp[idx], d["f1"][idx]
    return f[a & ~im].sum() - (a & im).sum(), (~a & ~im).sum()


def gstar(m, idx):
    a1, r1 = parts(C[(m, "cite")], idx)
    a2, r2 = parts(C[(m, "cascade")], idx)
    if a1 <= a2:
        return 0.0
    if r1 <= r2:
        return math.inf
    return (a1 - a2) / (r1 - r2)


print("\n== 4. Crossover gamma* (cite beats cascade iff gamma < gamma*; beta=1), bootstrap B=2000 ==")
rng = np.random.default_rng(0)
boots = [rng.integers(0, 300, 300) for _ in range(2000)]
full = np.arange(300)
for m in M:
    g = gstar(m, full)
    gb = np.array([gstar(m, b) for b in boots])
    lo = np.percentile(gb, 2.5)
    hi = np.percentile(np.where(np.isinf(gb), 1e9, gb), 97.5)
    a1, r1 = parts(C[(m, "cite")], full)
    a2, r2 = parts(C[(m, "cascade")], full)
    his = "inf" if hi >= 1e8 else f"{hi:.3f}"
    print(f"{m:<7} gamma*={g:.3f}  95% CI [{lo:.3f}, {his}]  P(g*>0.5)={np.mean(gb > 0.5):.2f}  "
          f"P(g*>1)={np.mean(gb > 1):.2f} | cite A={a1:.1f} FR={r1}  cascade A={a2:.1f} FR={r2}")

print("\n== 5. Generation determinism: 30 diagnostic keys generated twice ==")
if os.path.exists("outputs_v3_diag/generations.jsonl"):
    D = {g["key"]: g for g in R("outputs_v3_diag/generations.jsonl")}
    F = {g["key"]: g for g in R(f"{OUT}/generations.jsonl")}
    print(f"same outcome {sum(D[k]['cite_outcome'] == F[k]['cite_outcome'] for k in D)}/30, "
          f"identical answer {sum(D[k]['answer'] == F[k]['answer'] for k in D)}/30")
else:
    print("  skipped (outputs_v3_diag/ not found)")

print("\n== 6. False-answer sources (script 11) ==")
src = json.load(open(f"{OUT}/false_answer_sources.json"))
for k in ["BM25|cite", "Dense|cite", "Hybrid|cite", "BM25|explicit", "Dense|explicit", "Hybrid|explicit",
          "BM25|naive", "Dense|naive", "Hybrid|naive"]:
    s = src[k]
    print(f"{k:<16} FA={s['n_false_answers']:>3}  designated={s['sources'].get('designated', 0):>3} "
          f"other={s['sources'].get('other_passage', 0):>3}  FAR|hit={s['far_designated_hit']:.3f} (n={s['n_hit']})"
          f"  FAR|miss={s['far_designated_miss']:.3f} (n={s['n_miss']})")

print("\n== 7. Retrieval ==")
for m in M:
    print(m, S["retrieval"][m])

print("\n== 8. lambda* : U_rel(lambda) = (sum F1_CA + #CAbs - lambda #FA)/N ; explicit beats cite iff lambda > lambda* ==")
def ab(d, idx):
    a, im, f = d["a"][idx], imp[idx], d["f1"][idx]
    return f[a & ~im].sum() + (~a & im).sum(), (a & im).sum()
for m in M:
    def ls(idx):
        g1, f1_ = ab(C[(m, "cite")], idx); g2, f2_ = ab(C[(m, "explicit")], idx)
        if f1_ <= f2_: return math.inf if g1 >= g2 else 0.0
        return max(0.0, (g1 - g2) / (f1_ - f2_))
    l = ls(full); lb = np.array([ls(b) for b in boots])
    hi = np.percentile(np.where(np.isinf(lb), 1e9, lb), 97.5)
    print(f"{m:<7} lambda*={l:.2f}  95% CI [{np.percentile(lb, 2.5):.2f}, {'inf' if hi >= 1e8 else f'{hi:.2f}'}]")

"""Step 13 — v3 result figures (offline, free).

Recomputes everything from <results>/generations.jsonl (no stale JSON), so the plotted
numbers are the same ones as in scripts/12_rigor_checks.py.

  far_f1_scatter.pdf   FAR (Wilson 95% CI) vs F1 over all answerable questions, 9 configs
  tradeoff_bm25.pdf    BM25: (a) U_rel vs lambda (cost of a false answer)
                             (b) U_biz vs gamma (cost of a missed answerable question, beta=1)

    python scripts/13_figures_v3.py            # writes both sizes
    python scripts/13_figures_v3.py --report   # small (8 pt) versions only
Small (8 pt, paper column width) -> figures/small/ ; large fonts (posters, slides) -> figures/large/
"""
import json
import math
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from raglib import abstention, metrics

RES = ROOT / os.environ.get("RAG_OUTPUT_SUBDIR", "outputs_v3")
OUT_REPORT = ROOT / "figures" / "small"
OUT_POSTER = ROOT / "figures" / "large"
M = ["BM25", "Dense", "Hybrid"]
V = ["naive", "explicit", "cite"]
COLOR = {"naive": "#D55E00", "explicit": "#0072B2", "cite": "#009E73", "cascade": "#CC79A7"}  # Okabe-Ito
STYLE = {"naive": ":", "explicit": "--", "cite": "-", "cascade": "-."}
MARK = {"BM25": "o", "Dense": "s", "Hybrid": "^"}
INK, MUTED = "#222222", "#666666"


def load():
    R = lambda p: [json.loads(l) for l in open(p, encoding="utf-8")]
    Q = {x["qid"]: x for x in R(ROOT / "data" / "queries.jsonl")}
    qids = sorted(Q)
    G = {(g["method"], g["variant"], g["qid"]): g["answer"] for g in R(RES / "generations.jsonl")}
    imp = np.array([Q[i]["is_impossible"] for i in qids])
    gold = [Q[i]["gold_answer"] for i in qids]

    def vec(texts):
        a = np.array([bool(t.strip()) and not abstention.is_abstention(t) for t in texts])
        f1 = np.array([metrics.squad_f1(t, g) if (x and not im) else 0.0
                       for t, g, x, im in zip(texts, gold, a, imp)])
        return a, f1

    C = {}
    for m in M:
        for v in V:
            C[(m, v)] = vec([G[(m, v, i)] for i in qids])
        casc = []
        for i in qids:
            c, e = G[(m, "cite", i)], G[(m, "explicit", i)]
            casc.append(c if not abstention.is_abstention(c) else
                        (e if not abstention.is_abstention(e) else "I don't know."))
        C[(m, "cascade")] = vec(casc)
    return imp, C


def wilson(k, n, z=1.96):
    p = k / n
    c = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    d = 1 + z * z / n
    return (p + z * z / (2 * n) - c) / d, (p + z * z / (2 * n) + c) / d


def style(poster):
    base = 20 if poster else 8
    plt.rcParams.update({
        "font.size": base, "axes.labelsize": base + (2 if poster else 0),
        "xtick.labelsize": base - (2 if poster else 1), "ytick.labelsize": base - (2 if poster else 1),
        "legend.fontsize": base - (2 if poster else 1), "lines.linewidth": 3 if poster else 1.5,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.spines.top": False, "axes.spines.right": False, "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "savefig.bbox": "tight",
        "pdf.fonttype": 42, "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.8})


def fig_scatter(imp, C, path, poster, ieee8=False):
    """FAR vs F1 for the 9 configurations. Legends sit outside the axes (right), nothing is
    printed on the data, and the 'better' direction is written into the axis labels.
    Same design for report and poster; only sizes differ. ieee8: all text 8 pt and math
    in Arial (IEEE asks for 8-10 pt at final size), used for the report."""
    style(poster)
    if ieee8:
        plt.rcParams.update({"axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
                             "legend.fontsize": 8, "legend.title_fontsize": 8,
                             "mathtext.fontset": "custom", "mathtext.rm": "Arial",
                             "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold"})
    # report: the axes are narrower so that axes + outside legend fill one IEEE column
    # ieee8: narrower axes so that axes + 8 pt legend are ~250 pt wide (one column at
    # natural size, no scaling, so the text stays 8 pt)
    fig, ax = plt.subplots(figsize=(8.4, 5.2) if poster else ((2.53, 2.35) if ieee8 else (2.75, 2.35)))
    ms = 15 if poster else 5.5
    na, nu = (~imp).sum(), imp.sum()
    for v in V:
        for m in M:
            a, f1 = C[(m, v)]
            fa = int((a & imp).sum())
            far, (lo, hi) = fa / nu, wilson(fa, nu)
            x = f1[~imp].sum() / na
            ax.errorbar(x, far, yerr=[[far - lo], [hi - far]], fmt=MARK[m], color=COLOR[v], ms=ms,
                        mec="white", mew=1.2 if poster else 0.5, ecolor=COLOR[v],
                        elinewidth=1.5 if poster else 0.8, capsize=0, alpha=0.95, zorder=3)
    if poster:
        ax.set_xlabel("F1 on answerable questions   (higher is better →)")
    else:
        ax.set_xlabel("F1 on answerable questions\n(higher is better →)")
    ax.set_ylabel("False-answer rate\n(lower is better ↓)")
    ax.set_xlim(0.30, 0.70); ax.set_ylim(0.0, 0.37)
    if not poster:
        ax.set_xticks([0.3, 0.4, 0.5, 0.6, 0.7])
    kw = dict(frameon=False, handletextpad=0.3, alignment="left", loc="upper left",
              borderaxespad=0 if not poster else 0.5)
    leg1 = ax.legend(handles=[Line2D([0], [0], marker="o", ls="", color=COLOR[v], ms=ms * 0.8, label=v)
                              for v in V],
                     title="Prompt", bbox_to_anchor=(1.02, 1.0), **kw)
    ax.add_artist(leg1)
    ax.legend(handles=[Line2D([0], [0], marker=MARK[m], ls="", color=MUTED, ms=ms * 0.8, label=m)
                       for m in M],
              title="Retriever", bbox_to_anchor=(1.02, 0.50), **kw)
    for lg in ax.get_legend(), leg1:
        lg.get_title().set_fontweight("bold")
    fig.savefig(path); plt.close(fig)


def fig_tradeoff(imp, C, path, poster, m="BM25"):
    style(poster)
    # poster: smaller canvas = larger fonts once scaled to the column width
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.0, 5.6) if poster else (7.0, 2.35))
    N = len(imp)

    def parts(key):
        a, f1 = C[(m, key)]
        return f1[a & ~imp].sum(), int((~a & imp).sum()), int((a & imp).sum()), int((~a & ~imp).sum())

    lam = np.linspace(0, 6, 121)
    # (a) also shows the cascade: it sits below cite for every lambda
    for v in V + ["cascade"]:
        s, cab, fa, _ = parts(v)
        y = (s + cab - lam * fa) / N
        a1.plot(lam, y, STYLE[v], color=COLOR[v], label=v)
    sc, cc, fc, _ = parts("cite"); se, ce, fe, _ = parts("explicit")
    ls = ((sc + cc) - (se + ce)) / (fc - fe)
    a1.axvline(ls, color=MUTED, lw=1, ls=(0, (2, 2)))
    a1.annotate(f"λ* = {ls:.1f}", (ls, a1.get_ylim()[1]), xytext=(4, -4), textcoords="offset points",
                va="top", color=MUTED)
    a1.set_xlabel("λ = cost of a false answer" if not poster else "λ (in correct answers)")
    a1.set_ylabel("$U_{rel}$" if not poster else "Utility U(λ)")
    a1.set_title("(a) reliability utility" if not poster else "(a) false answers cost λ",
                 loc="left", color=INK)

    gam = np.linspace(0, 4, 81)
    for v in ["naive", "explicit", "cite", "cascade"]:
        s, _, fa, fr = parts(v)
        y = (s - fa - gam * fr) / N
        a2.plot(gam, y, STYLE[v], color=COLOR[v], label=v)
    s1, _, f1_, r1 = parts("cite"); s2, _, f2_, r2 = parts("cascade")
    gs = ((s1 - f1_) - (s2 - f2_)) / (r1 - r2)
    a2.axvline(gs, color=MUTED, lw=1, ls=(0, (2, 2)))
    a2.annotate(f"γ* = {gs:.1f}", (gs, a2.get_ylim()[0]), xytext=(-4, 4) if poster else (4, 4),
                textcoords="offset points", va="bottom", ha="right" if poster else "left", color=MUTED)
    a2.set_xlabel("γ = cost of a missed question (β = 1)" if not poster else "γ (in correct answers)")
    a2.set_ylabel("$U_{biz}$" if not poster else "Utility U(γ)")
    a2.set_title("(b) business utility" if not poster else "(b) missed questions cost γ",
                 loc="left", color=INK)
    for ax in (a1, a2):
        ax.margins(x=0.02)
    h, l = a2.get_legend_handles_labels()            # one shared legend below both panels
    if not poster:
        fig.tight_layout(w_pad=2.5, rect=(0, 0.11, 1, 1))
        fig.legend(h, l, loc="lower center", ncol=4, frameon=False, handlelength=2.6, columnspacing=2.0)
        fig.savefig(path); plt.close(fig)
        return
    # poster: put the legend a fixed small gap below the x-axis labels
    # (0.11 in in the figure ~ 0.2 cm of white space on the printed poster)
    fig.tight_layout(w_pad=1.5, rect=(0, 0.075, 1, 1))
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    label_bottom = min(ax.xaxis.label.get_window_extent(r).y0 for ax in (a1, a2)) / fig.dpi
    fig.legend(h, l, loc="upper center", ncol=4, frameon=False, handlelength=2.6, columnspacing=2.0,
               borderpad=0, borderaxespad=0,
               bbox_to_anchor=(0.5, (label_bottom - 0.11) / fig.get_figheight()))
    fig.savefig(path, pad_inches=0.01); plt.close(fig)


def fig_tradeoff_stacked(imp, C, path, m="BM25"):
    """Report, one column: the poster's tradeoff figure with (a) above (b).
    Follows the IEEE TAI template: axis labels use words plus the symbol, all text is
    8 pt at final size (the report includes the PDF at natural size), and the subfigure
    labels (a), (b) are centered below each panel in 8 pt Times. Tall panels (as on the
    poster) spread the lines apart vertically; the shared legend sits on top, so it is
    read before the data. The panel descriptions are in the LaTeX caption."""
    style(False)
    plt.rcParams.update({"axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "legend.fontsize": 8, "axes.titlesize": 8,
                         # math text in Arial too (IEEE accepts Times/Helvetica/Arial/...)
                         "mathtext.fontset": "custom", "mathtext.rm": "Arial",
                         "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold"})
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(3.0, 5.1))
    N = len(imp)

    def parts(key):
        a, f1 = C[(m, key)]
        return f1[a & ~imp].sum(), int((~a & imp).sum()), int((a & imp).sum()), int((~a & ~imp).sum())

    lam = np.linspace(0, 6, 121)
    for v in V + ["cascade"]:
        s, cab, fa, _ = parts(v)
        a1.plot(lam, (s + cab - lam * fa) / N, STYLE[v], color=COLOR[v], label=v)
    sc, cc, fc, _ = parts("cite"); se, ce, fe, _ = parts("explicit")
    ls = ((sc + cc) - (se + ce)) / (fc - fe)
    a1.axvline(ls, color=MUTED, lw=1, ls=(0, (2, 2)))
    a1.annotate(f"λ* = {ls:.1f}", (ls, a1.get_ylim()[1]), xytext=(4, -4), textcoords="offset points",
                va="top", color=MUTED)
    a1.set_xlabel("Cost of a false answer λ (correct answers)")
    a1.set_ylabel("Utility $U_{\\mathrm{rel}}$")

    gam = np.linspace(0, 4, 81)
    for v in ["naive", "explicit", "cite", "cascade"]:
        s, _, fa, fr = parts(v)
        a2.plot(gam, (s - fa - gam * fr) / N, STYLE[v], color=COLOR[v], label=v)
    s1, _, f1_, r1 = parts("cite"); s2, _, f2_, r2 = parts("cascade")
    gs = ((s1 - f1_) - (s2 - f2_)) / (r1 - r2)
    a2.axvline(gs, color=MUTED, lw=1, ls=(0, (2, 2)))
    a2.annotate(f"γ* = {gs:.1f}", (gs, a2.get_ylim()[0]), xytext=(-4, 4), textcoords="offset points",
                va="bottom", ha="right", color=MUTED)
    a2.set_xlabel("Cost of a missed question γ (correct answers)")
    a2.set_ylabel("Utility $U_{\\mathrm{biz}}$")
    from matplotlib.ticker import MultipleLocator
    for ax in (a1, a2):
        ax.margins(x=0.02)
        ax.yaxis.set_major_locator(MultipleLocator(0.1))   # fixed fine grid: the tall panels
        ax.xaxis.set_major_locator(MultipleLocator(1))     # exist to show these details
    h, l = a2.get_legend_handles_labels()            # one shared legend above both panels
    fig.tight_layout(h_pad=2.6, rect=(0, 0.035, 1, 0.972))  # room for the (a)/(b) labels
    fig.legend(h, l, loc="upper center", ncol=4, frameon=False, handlelength=2.4, columnspacing=1.0,
               handletextpad=0.4, borderaxespad=0.1)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    for ax, lab in ((a1, "(a)"), (a2, "(b)")):     # centered below each panel's x label
        pos = ax.get_position()
        y = (ax.xaxis.label.get_window_extent(r).y0 / fig.dpi - 0.05) / fig.get_figheight()
        fig.text((pos.x0 + pos.x1) / 2, y, lab, ha="center", va="top", color=INK,
                 fontfamily="Times New Roman", fontsize=8)
    fig.savefig(path); plt.close(fig)


def main():
    imp, C = load()
    if "--report" in sys.argv:                        # report figures only, IEEE 8 pt text
        OUT_REPORT.mkdir(parents=True, exist_ok=True)
        fig_scatter(imp, C, OUT_REPORT / "far_f1_scatter.pdf", False, ieee8=True)
        fig_tradeoff_stacked(imp, C, OUT_REPORT / "tradeoff_bm25_stacked.pdf")
        prev = os.environ.get("PREVIEW_DIR")
        if prev:
            fig_scatter(imp, C, Path(prev) / "far_f1_scatter_report8.png", False, ieee8=True)
        print(f"[13] wrote far_f1_scatter.pdf (8 pt), tradeoff_bm25_stacked.pdf -> {OUT_REPORT}")
        return
    if "--stacked" in sys.argv:                       # only the one-column report version
        OUT_REPORT.mkdir(parents=True, exist_ok=True)
        fig_tradeoff_stacked(imp, C, OUT_REPORT / "tradeoff_bm25_stacked.pdf")
        prev = os.environ.get("PREVIEW_DIR")
        if prev:
            fig_tradeoff_stacked(imp, C, Path(prev) / "tradeoff_bm25_stacked.png")
        print(f"[13] wrote tradeoff_bm25_stacked.pdf -> {OUT_REPORT}")
        return
    OUT_REPORT.mkdir(parents=True, exist_ok=True); OUT_POSTER.mkdir(parents=True, exist_ok=True)
    for poster, out in ((False, OUT_REPORT), (True, OUT_POSTER)):
        fig_scatter(imp, C, out / "far_f1_scatter.pdf", poster)
        fig_tradeoff(imp, C, out / "tradeoff_bm25.pdf", poster)
        prev = os.environ.get("PREVIEW_DIR")             # optional PNG previews for checking by eye
        if prev:
            tag = "poster" if poster else "report"
            fig_scatter(imp, C, Path(prev) / f"far_f1_scatter_{tag}.png", poster)
            fig_tradeoff(imp, C, Path(prev) / f"tradeoff_bm25_{tag}.png", poster)
        print(f"[13] wrote far_f1_scatter.pdf, tradeoff_bm25.pdf -> {out}")


if __name__ == "__main__":
    main()

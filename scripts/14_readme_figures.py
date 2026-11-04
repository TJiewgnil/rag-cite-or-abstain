"""Step 14: README figures in English and Chinese (offline, free).

Uses the same computation as scripts/13_figures_v3.py (recomputed from
outputs_v3/generations.jsonl), so the plotted numbers match the tables.

  docs/images/far-f1-en.png / far-f1-zh.png        false-answer rate vs F1, 9 configurations
  docs/images/tradeoff-en.png / tradeoff-zh.png    BM25: utility vs the cost of a false answer

    python scripts/14_readme_figures.py
"""
import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("fig13", ROOT / "scripts" / "13_figures_v3.py")
f13 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(f13)
M, V, COLOR, STYLE, MARK, INK, MUTED = f13.M, f13.V, f13.COLOR, f13.STYLE, f13.MARK, f13.INK, f13.MUTED
OUT = ROOT / "docs" / "images"

TEXT = {
    "en": {
        "font": ["Arial", "DejaVu Sans"],
        "x": "F1 on answerable questions (higher is better)",
        "y": "False-answer rate (lower is better)",
        "prompt": "Prompt", "retriever": "Retriever",
        "names": {"naive": "naive", "explicit": "explicit abstain", "cite": "cite-or-abstain"},
        "title": "Each point: one retriever × prompt, 300 questions (bars: 95% CI)",
        "lam_x": "Cost of one false answer, in correct answers (λ)",
        "lam_y": "Utility per question",
        "lam_title": "BM25: which prompt is better depends on what a false answer costs",
        "cross": "explicit wins above λ = {v:.1f}",
    },
    "zh": {
        "font": ["Microsoft YaHei", "Noto Sans SC", "SimHei", "DejaVu Sans"],
        "x": "可答题上的 F1（越高越好）",
        "y": "错误回答率（越低越好）",
        "prompt": "提示", "retriever": "检索器",
        "names": {"naive": "直接回答", "explicit": "明确拒答", "cite": "有引用才回答"},
        "title": "每个点：一种检索器 × 提示组合，300 道题（竖线：95% 置信区间）",
        "lam_x": "一次错误回答的代价，折合正确回答数（λ）",
        "lam_y": "每题效用",
        "lam_title": "BM25：哪种提示更好，取决于错误回答的代价",
        "cross": "λ > {v:.1f} 时明确拒答更优",
    },
}


def setup(lang):
    f13.style(poster=False)
    plt.rcParams.update({"font.size": 11, "axes.labelsize": 11, "xtick.labelsize": 10,
                         "ytick.labelsize": 10, "legend.fontsize": 10, "lines.linewidth": 2,
                         "font.sans-serif": TEXT[lang]["font"], "axes.unicode_minus": False,
                         "figure.facecolor": "white", "savefig.facecolor": "white", "savefig.dpi": 160})


def far_f1(imp, C, lang):
    t = TEXT[lang]
    setup(lang)
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    na, nu = (~imp).sum(), imp.sum()
    for v in V:
        for m in M:
            a, f1 = C[(m, v)]
            fa = int((a & imp).sum())
            far, (lo, hi) = fa / nu, f13.wilson(fa, nu)
            x = f1[~imp].sum() / na
            ax.errorbar(x, far, yerr=[[far - lo], [hi - far]], fmt=MARK[m], color=COLOR[v], ms=8,
                        mec="white", mew=0.8, ecolor=COLOR[v], elinewidth=1.2, capsize=0, zorder=3)
    ax.set_xlabel(t["x"]); ax.set_ylabel(t["y"])
    ax.set_xlim(0.30, 0.70); ax.set_ylim(0.0, 0.37)
    ax.set_title(t["title"], fontsize=10, color=MUTED, loc="left")
    kw = dict(frameon=False, loc="upper left", handletextpad=0.3)
    leg1 = ax.legend(handles=[Line2D([0], [0], marker="o", ls="", color=COLOR[v], ms=8, label=t["names"][v])
                              for v in V], title=t["prompt"], bbox_to_anchor=(1.01, 1.0), **kw)
    ax.add_artist(leg1)
    leg2 = ax.legend(handles=[Line2D([0], [0], marker=MARK[m], ls="", color=MUTED, ms=8, label=m) for m in M],
                     title=t["retriever"], bbox_to_anchor=(1.01, 0.52), **kw)
    for lg in (leg1, leg2):
        lg.get_title().set_fontweight("bold")
    fig.savefig(OUT / f"far-f1-{lang}.png", bbox_inches="tight", pad_inches=0.15,
                bbox_extra_artists=[leg1, leg2])  # keep both outside legends inside the image
    plt.close(fig)


def tradeoff(imp, C, lang, m="BM25"):
    t = TEXT[lang]
    setup(lang)
    N = len(imp)
    lam = [i / 10 for i in range(0, 81)]
    parts = {}
    for v in V:
        a, f1 = C[(m, v)]
        s = f1[~imp].sum()                 # F1 summed over answered answerable questions
        cab = int((~a & imp).sum())        # correct abstentions
        fa = int((a & imp).sum())          # false answers
        parts[v] = (s, cab, fa)
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    for v in V:
        s, cab, fa = parts[v]
        ax.plot(lam, [(s + cab - l * fa) / N for l in lam], STYLE[v], color=COLOR[v], label=t["names"][v])
    (sc, cc, fc), (se, ce, fe) = parts["cite"], parts["explicit"]
    x = ((sc + cc) - (se + ce)) / (fc - fe)    # where cite and explicit have equal utility
    y = (sc + cc - x * fc) / N
    ax.axvline(x, color=MUTED, lw=1, ls=(0, (2, 2)))
    ax.annotate(t["cross"].format(v=x), (x, y), xytext=(x + 0.4, y + 0.06), fontsize=10, color=INK,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.set_xlabel(t["lam_x"]); ax.set_ylabel(t["lam_y"])
    ax.set_title(t["lam_title"], fontsize=10, color=MUTED, loc="left")
    ax.legend(frameon=False, loc="lower left")
    fig.savefig(OUT / f"tradeoff-{lang}.png", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return x


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    imp, C = f13.load()
    for lang in ("en", "zh"):
        far_f1(imp, C, lang)
        x = tradeoff(imp, C, lang)
    print(f"[14] wrote far-f1 and tradeoff figures (en, zh) -> {OUT}; BM25 crossover lambda = {x:.2f}")


if __name__ == "__main__":
    main()

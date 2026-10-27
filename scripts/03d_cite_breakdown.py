"""Step 03d — why did cite abstain? (review item C1). Offline, free.

Splits every cite output into four outcomes, per retriever and answerability:
  answered          verified quote -> answer used
  model_declined    the model said has_answer=false (the designed abstention)
  quote_unverified  the model answered but its quote is not in the retrieved context
  format_failure    reply missing / not a valid JSON object (engineering failure)
For v2 rows (no raw reply saved) the last class is "format_failure_suspected".

It also reports FAR and answerable coverage computed only on outputs with a valid
model decision, so the paper can state how much of the reliability gain survives
once format failures are taken out.

    python scripts/03d_cite_breakdown.py                         # current RAG_OUTPUT_SUBDIR
    RAG_OUTPUT_SUBDIR=outputs_v3 python scripts/03d_cite_breakdown.py

Writes <output dir>/cite_breakdown.json.
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from raglib import config, data, generator

ORDER = ["answered", "model_declined", "quote_unverified",
         "format_failure", "format_failure_suspected"]


def main():
    q = {r["qid"]: r for r in data.read_jsonl(config.QUERIES_PATH)}
    rows = [g for g in data.read_jsonl(config.GENERATIONS_PATH) if g["variant"] == "cite"]
    if not rows:
        sys.exit(f"no cite rows in {config.GENERATIONS_PATH}")
    out = {}
    print(f"cite outcomes in {config.OUTPUT_DIR.name}/ (n per cell)")
    print(f"{'method':<8}{'question':<14}" + "".join(f"{c:>26}" for c in ORDER))
    for m in sorted({g["method"] for g in rows}):
        out[m] = {}
        for imp, label in ((False, "answerable"), (True, "unanswerable")):
            sub = [g for g in rows if g["method"] == m and bool(q[g["qid"]]["is_impossible"]) == imp]
            c = Counter(generator.infer_cite_outcome(g) for g in sub)
            out[m][label] = {**{k: c.get(k, 0) for k in ORDER}, "n": len(sub)}
            print(f"{m:<8}{label:<14}" + "".join(f"{c.get(k, 0):>26}" for k in ORDER))
        a, u = out[m]["answerable"], out[m]["unanswerable"]
        fails = lambda d: d["format_failure"] + d["format_failure_suspected"]
        a_valid, u_valid = a["n"] - fails(a), u["n"] - fails(u)
        out[m]["all_outputs"] = {"coverage": a["answered"] / a["n"], "far": u["answered"] / u["n"]}
        out[m]["valid_decisions_only"] = {
            "coverage": a["answered"] / a_valid if a_valid else None,
            "far": u["answered"] / u_valid if u_valid else None,
            "n_answerable": a_valid, "n_unanswerable": u_valid}
    print("\nFAR / answerable coverage:  all outputs   vs   valid model decisions only")
    for m, d in out.items():
        v = d["valid_decisions_only"]
        cov = f"{v['coverage']:.3f}" if v["coverage"] is not None else "n/a"
        far = f"{v['far']:.3f}" if v["far"] is not None else "n/a"
        print(f"  {m:<7} FAR {d['all_outputs']['far']:.3f} -> {far}   "
              f"coverage {d['all_outputs']['coverage']:.3f} -> {cov}")
    fr = Counter(g.get("finish_reason", "not logged (v2)") for g in rows)
    ps = Counter(g.get("parse_status", "not logged (v2)") for g in rows)
    print(f"\nfinish_reason: {dict(fr)}\nparse_status:  {dict(ps)}")
    out["finish_reason"], out["parse_status"] = dict(fr), dict(ps)
    path = config.OUTPUT_DIR / "cite_breakdown.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[03d] wrote {path}")


if __name__ == "__main__":
    main()

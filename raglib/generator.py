"""Generation: GeminiGenerator (Vertex AI) + MockGenerator (0-cost smoke).

Design note (engineering discipline): generation is the only step that costs
money/time, so it is decoupled from evaluation. `generate_batch` caches every
answer to a jsonl keyed by (method, variant, qid); re-runs skip cached keys, so
you generate once and then re-evaluate offline for free.

v3 change (cite variant): every cite reply is now saved with its raw text, finish
reason and a `cite_outcome` label (answered / model_declined / quote_unverified /
format_failure). In v2 an unparseable reply fell back to has_answer=True with an
empty quote and was silently counted as an abstention; that path produced about
half of all cite "abstentions" and could not be audited because the raw reply was
not stored.
"""
import hashlib
import json
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List

from . import config


# ──────────────────────────────────────────────
# Prompt building (shared by both generators)
# ──────────────────────────────────────────────
def build_prompt(query: str, context_texts: List[str], variant: str = "naive") -> str:
    context = "\n\n".join(f"[{i+1}] {t}" for i, t in enumerate(context_texts))
    if variant == "cite":
        instruction = (
            "You answer questions STRICTLY from the context below; you have no outside knowledge. "
            "Reply with ONE JSON object and nothing else, with keys has_answer, evidence_quote, answer.\n"
            "Rules:\n"
            "- has_answer=true ONLY if you can copy one EXACT sentence from the context that, on its "
            "own, directly answers the question; put that verbatim sentence in evidence_quote.\n"
            "- If no single sentence directly answers it, set has_answer=false, evidence_quote=\"\", "
            "answer=\"\". A passage merely on the same topic does NOT count as an answer.\n"
            "- When unsure, choose has_answer=false — abstaining is better than guessing.\n"
            "- answer must be a short span copied from evidence_quote."
        )
        return f"{instruction}\n\nContext:\n{context}\n\nQuestion: {query}\n\nJSON:"
    if variant == "explicit":
        instruction = (
            "Answer the question using ONLY the information in the context below. "
            "You MUST reply with exactly \"I don't know\" if the context does not "
            "contain enough information to answer. Do not guess or use outside knowledge."
        )
    else:  # "naive"
        instruction = (
            "Answer the question using ONLY the information in the context below. "
            "If the answer is not in the context, reply \"I don't know\"."
        )
    return f"{instruction}\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"


def _stable_int(s: str) -> int:
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16)


# ──────────────────────────────────────────────
# Cite reply parsing + outcome labels
# ──────────────────────────────────────────────
# JSON schema for the cite reply (Vertex AI response_schema). Typing has_answer as a
# BOOLEAN stops the model from answering "false" as a string, which bool() reads as True.
CITE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "has_answer": {"type": "BOOLEAN"},
        "evidence_quote": {"type": "STRING"},
        "answer": {"type": "STRING"},
    },
    "required": ["has_answer", "evidence_quote", "answer"],
}

# Alternative key names the model sometimes uses; accepted but marked "repaired".
_HAS_KEYS = ("has_answer", "hasAnswer", "answerable")
_QUOTE_KEYS = ("evidence_quote", "evidenceQuote", "evidence", "quote")

# parse_status values. "empty"/"invalid" are FORMAT FAILURES, not model abstentions.
PARSE_OK, PARSE_REPAIRED, PARSE_EMPTY, PARSE_INVALID = "ok", "repaired", "empty", "invalid"
FORMAT_FAILURES = (PARSE_EMPTY, PARSE_INVALID)

# cite outcome categories (reported separately; see scripts/03d_cite_breakdown.py)
CITE_ANSWERED = "answered"              # verified quote -> the answer is used
CITE_DECLINED = "model_declined"        # model said has_answer=false (the designed abstention)
CITE_UNVERIFIED = "quote_unverified"    # model answered, but its quote is not in the context
CITE_FORMAT_FAIL = "format_failure"     # reply missing or not a valid JSON object
CITE_OUTCOMES = (CITE_ANSWERED, CITE_DECLINED, CITE_UNVERIFIED, CITE_FORMAT_FAIL)

ABSTAIN_TEXT = "I don't know."


def _to_bool(v):
    """(value, exact): exact=False when a non-boolean had to be interpreted."""
    if isinstance(v, bool):
        return v, True
    if isinstance(v, str) and v.strip().lower() in ("true", "yes"):
        return True, False
    return False, False          # "false", "no", "", None, numbers ... -> not an answer


def parse_cite_response(text: str) -> dict:
    """Parse the cite JSON into {has_answer, evidence_quote, answer, parse_status}.

    Never raises (it runs on paid output). An unparseable reply returns has_answer=False
    with parse_status "empty"/"invalid", so the caller can count it as a format failure
    instead of mistaking it for a model decision.
    """
    raw = (text or "").strip()
    fail = {"has_answer": False, "evidence_quote": "", "answer": ""}
    if not raw:
        return {**fail, "parse_status": PARSE_EMPTY}
    s = raw
    if s.startswith("```"):                              # strip ```json ... ``` fences
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s).strip()
    repaired = False
    try:
        obj = json.loads(s)
    except Exception:
        m = re.search(r"\{.*\}", s, re.S)                # JSON object wrapped in extra text
        try:
            obj = json.loads(m.group(0)) if m else None
            repaired = True
        except Exception:
            obj = None
    if isinstance(obj, list):                            # [ {...} ] -> first object
        obj = next((o for o in obj if isinstance(o, dict)), None)
        repaired = True
    if not isinstance(obj, dict):
        return {**fail, "parse_status": PARSE_INVALID}
    has_key = next((k for k in _HAS_KEYS if k in obj), None)
    if has_key is None:
        return {**fail, "parse_status": PARSE_INVALID}
    has_answer, exact = _to_bool(obj[has_key])
    quote_key = next((k for k in _QUOTE_KEYS if k in obj), None)
    repaired = (repaired or not exact or has_key != "has_answer"
                or quote_key not in (None, "evidence_quote"))
    return {"has_answer": has_answer,
            "evidence_quote": str(obj.get(quote_key, "") or "") if quote_key else "",
            "answer": str(obj.get("answer", "") or ""),
            "parse_status": PARSE_REPAIRED if repaired else PARSE_OK}


def _quote_in_context(quote: str, context_texts: List[str]) -> float:
    """Fraction of the quote's tokens present in any single context passage, in [0,1].
    The verifiable-citation check: did the model quote something that actually appears
    in the retrieved context? Catches fabricated/empty citations at generation time."""
    def toks(s):
        return set(w.lower().strip(".,!?;:'\"()[]") for w in (s or "").split() if w.strip())
    qt = toks(quote)
    if not qt:
        return 0.0
    return max((len(qt & toks(p)) / len(qt) for p in context_texts if p), default=0.0)


def _norm_text(s: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", (s or "").lower()).split())


def classify_cite(res: dict, qmatch: float) -> str:
    """Map a parsed cite reply + its quote-match score to one of the four outcomes."""
    if res.get("parse_status") in FORMAT_FAILURES:
        return CITE_FORMAT_FAIL
    if not res["has_answer"]:
        return CITE_DECLINED
    if not res["answer"].strip() or qmatch < config.CITE_QUOTE_MATCH_MIN:
        return CITE_UNVERIFIED
    return CITE_ANSWERED


def infer_cite_outcome(row: dict) -> str:
    """Outcome of a saved cite row.

    v3 rows carry `cite_outcome`. For v2 rows (raw reply not saved) it is inferred:
    has_answer=True with an empty quote can only come from the old parse fallback or a
    malformed reply, so it is labelled "format_failure_suspected".
    """
    if row.get("cite_outcome"):
        return row["cite_outcome"]
    if row.get("answer", "").strip() != ABSTAIN_TEXT:
        return CITE_ANSWERED
    if not row.get("has_answer"):
        return CITE_DECLINED
    if not (row.get("evidence_quote") or "").strip():
        return CITE_FORMAT_FAIL + "_suspected"
    return CITE_UNVERIFIED


# ──────────────────────────────────────────────
# Client-side rate limiter (keeps 03 under the Vertex per-minute quota)
# ──────────────────────────────────────────────
class _RateLimiter:
    """Thread-safe: spaces calls >= 60/rpm apart so concurrent workers together
    cannot exceed `rpm` requests per minute. rpm<=0 disables throttling."""

    def __init__(self, rpm: float):
        self.min_interval = 60.0 / rpm if rpm and rpm > 0 else 0.0
        self._lock = threading.Lock()
        self._next = 0.0

    def acquire(self) -> None:
        if self.min_interval <= 0:
            return
        with self._lock:                      # reserve a time slot (fast)
            slot = max(time.monotonic(), self._next)
            self._next = slot + self.min_interval
        wait = slot - time.monotonic()        # then sleep to it, outside the lock
        if wait > 0:
            time.sleep(wait)


# ──────────────────────────────────────────────
# Gemini (Vertex AI)
# ──────────────────────────────────────────────
class GeminiGenerator:
    SUPPORTED_MODELS = {
        "flash": "gemini-1.5-flash",   # cheapest
        "pro": "gemini-1.5-pro",
        "flash2": "gemini-2.0-flash",  # newer; use if 1.5 is retired
    }

    def __init__(self, project_id: str = config.GCP_PROJECT,
                 location: str = config.GCP_LOCATION,
                 model_alias: str = config.GEMINI_MODEL_ALIAS,
                 max_output_tokens: int = config.MAX_OUTPUT_TOKENS,
                 rpm: float = None):
        import vertexai
        from vertexai.generative_models import GenerativeModel, GenerationConfig

        vertexai.init(project=project_id, location=location)
        self.model_name = self.SUPPORTED_MODELS.get(model_alias, model_alias)
        self.model = GenerativeModel(self.model_name)
        self.gen_config = GenerationConfig(
            max_output_tokens=max_output_tokens,
            temperature=0.0,   # deterministic -> reproducible evaluation
        )
        # cite asks for structured output: JSON mode + typed schema. Bigger budget because
        # gemini-2.5 spends "thinking" tokens from the same output budget; the retry config
        # doubles it again for replies that came back empty/truncated.
        self.gen_config_json = self._json_config(GenerationConfig, config.CITE_MAX_OUTPUT_TOKENS)
        self.gen_config_json_retry = self._json_config(GenerationConfig,
                                                       2 * config.CITE_MAX_OUTPUT_TOKENS)
        # Shared across worker threads in generate_batch -> global rate cap.
        self._limiter = _RateLimiter(config.GEMINI_RPM if rpm is None else rpm)

    @staticmethod
    def _json_config(GenerationConfig, max_tokens: int):
        """JSON mode + typed schema; plain JSON mode on SDKs without response_schema."""
        try:
            if not config.CITE_USE_SCHEMA:
                raise TypeError("disabled by CITE_USE_SCHEMA=0")
            return GenerationConfig(max_output_tokens=max_tokens, temperature=0.0,
                                    response_mime_type="application/json",
                                    response_schema=CITE_SCHEMA)
        except TypeError:
            print("  [generator] SDK has no response_schema; using plain JSON mode")
            return GenerationConfig(max_output_tokens=max_tokens, temperature=0.0,
                                    response_mime_type="application/json")

    def _generate_raw(self, prompt: str, gen_config, max_retries: int = 8):
        """Return (text, finish_reason); finish_reason is e.g. STOP / MAX_TOKENS / SAFETY."""
        for attempt in range(max_retries):
            self._limiter.acquire()                     # stay under the quota
            try:
                resp = self.model.generate_content(prompt, generation_config=gen_config)
            except Exception as e:                      # rate limit / transient -> retry
                if attempt == max_retries - 1:
                    raise
                # 429 / ResourceExhausted -> back off much longer; others short.
                msg = f"{e.__class__.__name__}: {e}"
                is_rate = ("429" in msg or "ResourceExhausted" in msg
                           or "Resource exhausted" in msg)
                base = 5.0 if is_rate else 1.0
                delay = min(base * (2 ** attempt), 60.0) + random.uniform(0, base)
                time.sleep(delay)                       # exponential backoff + jitter
                continue
            # A safety-blocked / MAX_TOKENS candidate may have no parts and resp.text
            # raises -> return "" and keep the finish reason so it can be diagnosed.
            try:
                fr = resp.candidates[0].finish_reason
                fr = getattr(fr, "name", str(fr))
            except Exception:
                fr = "UNKNOWN"
            try:
                return (resp.text or "").strip(), fr
            except Exception:
                return "", fr
        return "", "NO_RESPONSE"

    def _generate_text(self, prompt: str, gen_config, max_retries: int = 8) -> str:
        return self._generate_raw(prompt, gen_config, max_retries)[0]

    def generate(self, query: str, context_texts: List[str],
                 variant: str = "naive", max_retries: int = 8) -> str:
        return self._generate_text(build_prompt(query, context_texts, variant),
                                   self.gen_config, max_retries)

    def generate_meta(self, query: str, context_texts: List[str],
                      variant: str = "naive", max_retries: int = 8):
        """Like generate(), but returns (answer, finish_reason)."""
        return self._generate_raw(build_prompt(query, context_texts, variant),
                                  self.gen_config, max_retries)

    def generate_cite(self, query: str, context_texts: List[str],
                      max_retries: int = 8) -> dict:
        """Parsed cite reply + diagnostics (raw_response, finish_reason, attempts).
        An empty / non-JSON reply is re-asked CITE_PARSE_RETRIES times with double the
        token budget (truncation by thinking tokens is the usual cause)."""
        prompt = build_prompt(query, context_texts, "cite")
        raw, fr = self._generate_raw(prompt, self.gen_config_json, max_retries)
        res = parse_cite_response(raw)
        attempts = 1
        while res["parse_status"] in FORMAT_FAILURES and attempts <= config.CITE_PARSE_RETRIES:
            raw, fr = self._generate_raw(prompt, self.gen_config_json_retry, max_retries)
            res = parse_cite_response(raw)
            attempts += 1
        res.update(raw_response=raw, finish_reason=fr, attempts=attempts)
        return res


# ──────────────────────────────────────────────
# Mock generator — zero cost, deterministic, for smoke tests
# ──────────────────────────────────────────────
class MockGenerator:
    """Fakes a RAG model without any API call.

    Deterministically spreads outcomes across all four abstention cells so the
    evaluation logic can be exercised offline:
      - no context           -> abstain
      - hash(query) % 3 == 0 -> abstain  (plus, for 'explicit', % 3 == 1 too)
      - otherwise            -> echo the first sentence of the top passage
    """
    model_name = "mock"

    def generate(self, query: str, context_texts: List[str],
                 variant: str = "naive", **_) -> str:
        if not context_texts:
            return "I don't know."
        h = _stable_int(query) % 3
        abstain = (h == 0) or (variant == "explicit" and h == 1)
        if abstain:
            return "I don't know, the context does not contain the answer."
        first_sentence = re.split(r"(?<=[.!?])\s+", context_texts[0].strip())[0]
        return " ".join(first_sentence.split()[:25])

    def generate_meta(self, query: str, context_texts: List[str],
                      variant: str = "naive", **_):
        return self.generate(query, context_texts, variant), "STOP"

    def generate_cite(self, query: str, context_texts: List[str], **_) -> dict:
        if not context_texts or _stable_int(query) % 3 == 0:   # deterministic abstain
            raw = json.dumps({"has_answer": False, "evidence_quote": "", "answer": ""})
        else:
            first = re.split(r"(?<=[.!?])\s+", context_texts[0].strip())[0]
            raw = json.dumps({"has_answer": True, "evidence_quote": first,
                              "answer": " ".join(first.split()[:25])})
        res = parse_cite_response(raw)          # same code path as the real generator
        res.update(raw_response=raw, finish_reason="STOP", attempts=1)
        return res


# ──────────────────────────────────────────────
# Batch runner with caching + concurrency
# ──────────────────────────────────────────────
def cache_key(method: str, variant: str, qid: str) -> str:
    return f"{method}::{variant}::{qid}"


def cite_row(t: dict, res: dict, model_name: str) -> dict:
    """Build the cached row for one cite reply (kept separate so it can be unit-tested)."""
    qmatch = _quote_in_context(res["evidence_quote"], t["context_texts"])
    outcome = classify_cite(res, qmatch)
    ans = res["answer"].strip() if outcome == CITE_ANSWERED else ABSTAIN_TEXT
    return {"key": t["key"], "method": t["method"], "variant": t["variant"],
            "qid": t["qid"], "answer": ans, "model": model_name,
            "has_answer": res["has_answer"], "evidence_quote": res["evidence_quote"],
            "quote_match_gen": round(qmatch, 4),
            "answer_in_quote": bool(res["answer"].strip())
                               and _norm_text(res["answer"]) in _norm_text(res["evidence_quote"]),
            "cite_outcome": outcome,
            "parse_status": res.get("parse_status", PARSE_OK),
            "finish_reason": res.get("finish_reason", ""),
            "attempts": res.get("attempts", 1),
            "raw_response": res.get("raw_response", "")}


def generate_batch(generator, tasks: List[dict], cache_path,
                   concurrency: int = 1) -> List[dict]:
    """Run generation over `tasks`, caching results to `cache_path` (jsonl).

    Each task: {"key","method","variant","qid","question","context_texts"}.
    Already-cached keys are skipped. Returns all rows (cached + new).
    """
    cache_path = Path(cache_path)
    cached = {}
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    cached[row["key"]] = row

    todo = [t for t in tasks if t["key"] not in cached]

    def _run(t):
        if t["variant"] == "cite":
            res = generator.generate_cite(t["question"], t["context_texts"])
            return cite_row(t, res, generator.model_name)
        if hasattr(generator, "generate_meta"):
            answer, fr = generator.generate_meta(t["question"], t["context_texts"], t["variant"])
        else:
            answer, fr = generator.generate(t["question"], t["context_texts"], t["variant"]), ""
        return {"key": t["key"], "method": t["method"], "variant": t["variant"],
                "qid": t["qid"], "answer": answer, "model": generator.model_name,
                "finish_reason": fr}

    new_rows = []
    failed = 0
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "a", encoding="utf-8") as f:
        if concurrency <= 1:
            for t in todo:
                try:
                    row = _run(t)
                except Exception as e:                  # don't let one task kill the run
                    failed += 1
                    print(f"  ! skip {t['key']}: {e.__class__.__name__}: {e}")
                    continue
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                new_rows.append(row)
        else:
            with ThreadPoolExecutor(max_workers=concurrency) as ex:
                futures = {ex.submit(_run, t): t for t in todo}
                for fut in as_completed(futures):
                    try:
                        row = fut.result()
                    except Exception as e:
                        failed += 1
                        print(f"  ! skip {futures[fut]['key']}: "
                              f"{e.__class__.__name__}: {e}")
                        continue
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    f.flush()
                    new_rows.append(row)

    if failed:
        print(f"  [generate_batch] {failed} task(s) failed and were skipped "
              f"(not cached). Re-run the SAME command to retry only those.")
    n_fmt = sum(1 for r in new_rows if r.get("cite_outcome") == CITE_FORMAT_FAIL)
    if n_fmt:
        print(f"  [generate_batch] {n_fmt} cite repl(ies) were still not valid JSON after "
              f"retry -> counted as format_failure (see raw_response).")
    return list(cached.values()) + new_rows

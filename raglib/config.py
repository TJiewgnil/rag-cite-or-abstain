"""Central configuration: paths, model ids, hyperparameters.

Override the GCP settings with environment variables so you never hard-code your
project id into git:
    PowerShell:  $env:GCP_PROJECT = "my-project-123"
    bash (VM):   export GCP_PROJECT=my-project-123
"""
import os
from pathlib import Path

# ── Paths (resolve relative to rag_framework/, the package parent) ──
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
# RAG_OUTPUT_SUBDIR lets a new run live next to the old one without overwriting it,
# e.g. RAG_OUTPUT_SUBDIR=outputs_v3 for the cite re-run (the v2 results stay in outputs/).
OUTPUT_DIR = ROOT / os.environ.get("RAG_OUTPUT_SUBDIR", "outputs")
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Standard artefact locations (the pipeline steps read/write these)
CORPUS_PATH = DATA_DIR / "corpus.jsonl"
QUERIES_PATH = DATA_DIR / "queries.jsonl"
RETRIEVAL_PATH = OUTPUT_DIR / "retrieval.jsonl"
GENERATIONS_PATH = OUTPUT_DIR / "generations.jsonl"
GROUNDING_PATH = OUTPUT_DIR / "grounding.jsonl"

# ── Dataset ──
SQUAD_DATASET = "rajpurkar/squad_v2"

# ── Retrieval ──
TOP_K = 5
RRF_K = 60                                  # Reciprocal Rank Fusion constant
DENSE_MODEL = "BAAI/bge-small-en-v1.5"      # stronger retriever than MiniLM (MTEB); was sentence-transformers/all-MiniLM-L6-v2
# bge models want a short instruction prepended to the QUERY only (not documents).
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# ── Generation (Vertex AI Gemini) ──
GCP_PROJECT = os.environ.get("GCP_PROJECT", "your-project-id")
GCP_LOCATION = os.environ.get("GCP_LOCATION", "us-central1")
GEMINI_MODEL_ALIAS = os.environ.get("GEMINI_MODEL_ALIAS", "flash")
GEMINI_RPM = float(os.environ.get("GEMINI_RPM", "60"))   # client-side request cap (req/min); throttles 03 under the Vertex per-minute quota to avoid 429. Lower it (e.g. GEMINI_RPM=20) if you still get 429.
MAX_OUTPUT_TOKENS = 256
CITE_MAX_OUTPUT_TOKENS = 2048   # gemini-2.5 "thinking" tokens share the output budget; cite needs room for thoughts + JSON (was 1024 in v2)
CITE_USE_SCHEMA = os.environ.get("CITE_USE_SCHEMA", "1") != "0"   # typed response_schema for cite; set CITE_USE_SCHEMA=0 if the SDK rejects it
CITE_PARSE_RETRIES = 1          # re-ask once (with double the token budget) if the cite reply is not a valid JSON object
PROMPT_VARIANTS = ("naive", "explicit", "cite")  # ablation: plain / explicit-abstain / cite-or-abstain (structured)
CITE_QUOTE_MATCH_MIN = 0.5   # cite answers only if evidence_quote's tokens are >= this fraction present in a retrieved passage (verifiable citation)

# ── Answer metric ──
# "squad" = official SQuAD normalisation + counted overlap (v3 default).
# "legacy" = the v2 token_f1 (no normalisation); only for reproducing the v2 numbers.
F1_MODE = os.environ.get("F1_MODE", "squad")

# ── Grounding (NLI cross-encoder) ──
NLI_MODEL = "cross-encoder/nli-deberta-v3-small"
GROUNDING_THRESHOLD = 0.5                    # entailment prob >= this => claim supported


def get_device() -> str:
    """Return 'cuda' if a GPU is available, else 'cpu' (torch imported lazily)."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"

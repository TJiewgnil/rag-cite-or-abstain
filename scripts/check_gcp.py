"""Verify Vertex AI auth + Gemini model with ONE tiny call (the real S3 check).

    # PowerShell:  $env:GCP_PROJECT = "my-project-123"
    # VM bash:     export GCP_PROJECT=my-project-123
    python scripts/check_gcp.py

Run this once after enabling the Vertex AI API and `gcloud auth application-default
login`, BEFORE the full 03_generate run. Costs a fraction of a cent.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from raglib import config, generator


def main():
    print(f"project={config.GCP_PROJECT}  location={config.GCP_LOCATION}  "
          f"model_alias={config.GEMINI_MODEL_ALIAS}")
    if config.GCP_PROJECT == "your-project-id":
        print("! Set your project id first:")
        print("    PowerShell:  $env:GCP_PROJECT = 'your-real-project-id'")
        print("    VM bash:     export GCP_PROJECT=your-real-project-id")
        sys.exit(1)

    try:
        gen = generator.GeminiGenerator()
    except Exception as e:
        print(f"[FAIL] Vertex AI init: {e.__class__.__name__}: {e}")
        print("  -> enable the Vertex AI API in the console")
        print("  -> run: gcloud auth application-default login")
        sys.exit(1)

    try:
        ans = gen.generate("Reply with the single word: pong.",
                           ["(no context needed)"], variant="naive")
        print(f"[OK] model={gen.model_name} responded: {ans!r}")
        print("S3 (real) PASS -- safe to run: python scripts/03_generate.py --concurrency 12")
    except Exception as e:
        print(f"[FAIL] generate call: {e.__class__.__name__}: {e}")
        print("  -> model id may be retired; try: $env:GEMINI_MODEL_ALIAS = 'flash2'")
        sys.exit(1)


if __name__ == "__main__":
    main()

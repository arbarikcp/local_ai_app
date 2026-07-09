"""Lab 2 - attack RAG with a malicious document. Screens a real malicious
document (pulled from the red-team dataset) and a real clean document
through `screen_document_for_ingestion()` before they'd ever reach Module
11/12's ingestion pipeline - the poisoned document is quarantined, the
clean one passes through untouched.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "packages"))

from local_ai_core.security.rag_ingestion_guard import SourceTrust, screen_document_for_ingestion  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_PATH = REPO_ROOT / "datasets" / "red_team" / "red_team_prompts.jsonl"


def run_lab() -> dict:
    with DATASET_PATH.open(encoding="utf-8") as f:
        examples = [json.loads(line) for line in f if line.strip()]

    malicious_doc = next(e for e in examples if e["category"] == "indirect_injection_document" and e["is_malicious"])
    clean_doc = next(e for e in examples if e["surface"] == "uploaded_document" and not e["is_malicious"])

    malicious_decision = screen_document_for_ingestion(malicious_doc["text"], source_trust=SourceTrust.UNTRUSTED)
    clean_decision = screen_document_for_ingestion(clean_doc["text"], source_trust=SourceTrust.TRUSTED)

    return {
        "malicious_doc_text": malicious_doc["text"],
        "malicious_allowed": malicious_decision.allowed,
        "malicious_flagged_patterns": malicious_decision.flagged_patterns,
        "clean_doc_text": clean_doc["text"],
        "clean_allowed": clean_decision.allowed,
    }


def result_to_markdown(result: dict) -> str:
    return (
        "# Lab 2 - attack RAG with a malicious document\n\n"
        f"- Malicious document: \"{result['malicious_doc_text']}\"\n"
        f"  -> allowed: {result['malicious_allowed']} (flagged: {result['malicious_flagged_patterns']})\n"
        f"- Clean document: \"{result['clean_doc_text']}\"\n"
        f"  -> allowed: {result['clean_allowed']}\n"
    )


def main(argv: list[str] | None = None) -> int:
    result = run_lab()
    print(result_to_markdown(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

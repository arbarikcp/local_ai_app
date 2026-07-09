"""Labs 1-2 - load the real golden RAG dataset (16 hand-authored questions
over the Module 11 Nimbus handbook corpus) and demonstrate synthetic
question generation from a real document. Runs for real through loading
and summarizing; synthetic generation uses `FakeRuntime`, no live model.
"""

from __future__ import annotations

import asyncio
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "packages"))

from common import CORPUS_DIR, GOLDEN_SET_PATH  # noqa: E402
from local_ai_core.evals.golden_set import load_golden_set  # noqa: E402
from local_ai_core.evals.synthetic_questions import generate_questions_from_document  # noqa: E402
from local_ai_core.runtimes.fake import FakeRuntime  # noqa: E402
from local_ai_rag.loaders.markdown_loader import load_markdown_directory  # noqa: E402


def summarize_golden_set(cases) -> dict:
    return {
        "total": len(cases),
        "answerable": sum(1 for c in cases if not c.requires_refusal),
        "unanswerable": sum(1 for c in cases if c.requires_refusal),
        "by_difficulty": dict(Counter(c.difficulty for c in cases)),
        "by_category": dict(Counter(c.category for c in cases)),
    }


async def run_lab() -> dict:
    cases = load_golden_set(GOLDEN_SET_PATH)
    summary = summarize_golden_set(cases)

    documents = load_markdown_directory(CORPUS_DIR)
    sample_doc = next(d for d in documents if d.doc_id == "two_factor_authentication")
    runtime = FakeRuntime(
        default_response=(
            "How many backup codes does Nimbus provide for 2FA?\n"
            "Which plans require 2FA to be enabled?\n"
            "What happens if I lose my authenticator app and backup codes?"
        )
    )
    synthetic_questions = await generate_questions_from_document(sample_doc.text, runtime, model="fake-model", n=3)

    return {
        "golden_set_summary": summary,
        "sample_document": sample_doc.doc_id,
        "synthetic_questions": synthetic_questions,
    }


def result_to_markdown(result: dict) -> str:
    summary = result["golden_set_summary"]
    lines = ["# Labs 1-2 - golden RAG dataset and synthetic question generation\n"]
    lines.append(f"- Total golden cases: {summary['total']}")
    lines.append(f"- Answerable: {summary['answerable']}, Unanswerable: {summary['unanswerable']}")
    lines.append(f"- By difficulty: {summary['by_difficulty']}")
    lines.append(f"- By category: {summary['by_category']}")
    lines.append(f"\n## Synthetic questions generated from `{result['sample_document']}`\n")
    for q in result["synthetic_questions"]:
        lines.append(f"- {q}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    result = asyncio.run(run_lab())
    print(result_to_markdown(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

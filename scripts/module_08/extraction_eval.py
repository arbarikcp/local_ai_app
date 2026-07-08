"""Lab 7 — evaluate ExtractionPipeline against golden labels.

Reuses Module 3's golden extraction set directly (evals/golden_sets/
extraction.jsonl) rather than duplicating test data - this module doesn't
model every schema in that set (only PersonExtraction's name/age/city
shape), so evaluation is scoped to the matching subset, same as
constrained_decoding_runner.py.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "packages"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "module_01"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from constrained_decoding_runner import field_accuracy, load_matching_golden_cases  # noqa: E402
from local_ai_core.extraction.pipeline import ExtractionPipeline  # noqa: E402
from local_ai_core.extraction.review_queue import ReviewQueue  # noqa: E402
from local_ai_core.extraction.schemas import PersonExtraction  # noqa: E402


@dataclass(frozen=True)
class GoldenEvalCaseResult:
    case_id: str
    field_accuracy: float
    confidence: str
    needs_review: bool


@dataclass(frozen=True)
class GoldenEvalSummary:
    case_results: list[GoldenEvalCaseResult]
    review_queue: ReviewQueue

    @property
    def mean_field_accuracy(self) -> float:
        if not self.case_results:
            return 0.0
        return sum(r.field_accuracy for r in self.case_results) / len(self.case_results)

    @property
    def review_rate(self) -> float:
        if not self.case_results:
            return 0.0
        return sum(1 for r in self.case_results if r.needs_review) / len(self.case_results)


async def evaluate_against_golden_labels(
    pipeline: ExtractionPipeline, model: str, cases: list[dict]
) -> GoldenEvalSummary:
    case_results = []
    for case in cases:
        result = await pipeline.run(case["text"], model)
        accuracy = field_accuracy(result.fields, case["reference"])
        case_results.append(
            GoldenEvalCaseResult(
                case_id=case["id"], field_accuracy=accuracy, confidence=result.confidence, needs_review=result.needs_review
            )
        )
    return GoldenEvalSummary(case_results=case_results, review_queue=pipeline.review_queue)


def summary_to_markdown(summary: GoldenEvalSummary) -> str:
    header = "| Case | Field accuracy | Confidence | Needs review |\n|---|---:|---|---:|\n"
    rows = [
        f"| {r.case_id} | {r.field_accuracy:.0%} | {r.confidence} | {'yes' if r.needs_review else 'no'} |"
        for r in summary.case_results
    ]
    footer = (
        f"\nMean field accuracy: {summary.mean_field_accuracy:.0%}\n"
        f"Review rate: {summary.review_rate:.0%}\n"
        f"Pending human review: {len(summary.review_queue)} item(s)\n"
    )
    return header + "\n".join(rows) + "\n" + footer


def main(argv: list[str] | None = None) -> int:
    from ollama_probe import is_ollama_available
    from local_ai_core.runtimes.ollama import OllamaRuntime

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen2.5:1.5b")
    args = parser.parse_args(argv)

    if not is_ollama_available():
        print(
            "SKIPPED: Ollama is not reachable at http://localhost:11434. "
            "Install Ollama (Module 2) and re-run this lab on the resourced 32GB Mac.",
            file=sys.stderr,
        )
        return 1

    cases = load_matching_golden_cases()

    async def _run():
        runtime = OllamaRuntime()
        try:
            pipeline = ExtractionPipeline(runtime, PersonExtraction, required_fields=["name", "age", "city"])
            return await evaluate_against_golden_labels(pipeline, args.model, cases)
        finally:
            await runtime.aclose()

    summary = asyncio.run(_run())
    print(f"# Lab 7 — golden label evaluation\n\nModel: `{args.model}`\n")
    print(summary_to_markdown(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

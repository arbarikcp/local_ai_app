"""Evaluation command against a labeled dataset (PROPOSAL.md "How success
is measured", functional requirement 8). Same discipline as
`scripts/module_08/extraction_eval.py`'s golden-set harness, scoped to
this project's two schemas and its own dataset. Runs two real scenarios
against `FakeRuntime` (no model runtime on this machine): "perfect"
(proves the metrics correctly score a flawless run) and "imperfect"
(proves the metrics correctly catch a real, deliberately broken run) -
real model quality is honest-skip, deferred to the resourced 32GB Mac.
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "packages"))

from local_ai_core.deployment.config import AppConfig  # noqa: E402
from local_ai_core.runtimes.fake import FakeRuntime  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "schemas"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "prompts"))

from extraction_metrics import (  # noqa: E402
    EvalExample,
    field_exact_match,
    hallucinated_field_rate,
    missing_field_rate,
)
from extraction_service import ExtractionAppContext, build_extraction_context, run_extraction  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATASET_PATH = Path(__file__).resolve().parent / "extraction_dataset.jsonl"
CATALOG_PATH = REPO_ROOT / "models" / "MODEL_CATALOG.md"


def load_dataset(path: str | Path) -> list[EvalExample]:
    examples = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            examples.append(
                EvalExample(
                    id=record["id"], schema_name=record["schema_name"], text=record["text"], reference=record["reference"]
                )
            )
    return examples


class DatasetAwareRuntime(FakeRuntime):
    """Returns the reference JSON for whichever example's text is embedded
    in the prompt - a real, per-input-aware fake (matching Module 8's own
    `SequencedRuntime` precedent), not one canned response for every call.
    `corrupt_ids` deliberately returns invalid JSON for specific examples,
    to prove the eval harness's invalid-JSON/review-rate metrics catch a
    real failure rather than only ever seeing successes.
    """

    def __init__(self, examples: list[EvalExample], *, corrupt_ids: frozenset[str] = frozenset()) -> None:
        super().__init__()
        self._examples = examples
        self._corrupt_ids = corrupt_ids

    async def generate(self, request):
        matching = next((e for e in self._examples if e.text in request.prompt), None)
        if matching is None:
            text = "{}"
        elif matching.id in self._corrupt_ids:
            text = "not valid json at all"
        else:
            # InvoiceExtraction requires a model-self-reported `confidence`
            # field with no default (Module 8's own schema) even though
            # `compute_confidence()` explicitly never trusts that value
            # (confidence.py's own docstring) - real fields extracted from
            # the text live in `reference`; these two are schema-required
            # padding only, never scored (extraction_metrics.py only looks
            # at keys present in `reference`).
            payload = {**matching.reference, "confidence": "high", "evidence": {}}
            text = json.dumps(payload)
        self.responses = {request.model: text}
        return await super().generate(request)


@dataclass(frozen=True)
class EvalCaseResult:
    id: str
    schema_name: str
    field_exact_match: float
    missing_field_rate: float
    hallucinated_field_rate: float
    invalid_json: bool
    needs_review: bool
    latency_ms: float


@dataclass(frozen=True)
class EvalSummary:
    total: int
    mean_field_exact_match: float
    mean_missing_field_rate: float
    mean_hallucinated_field_rate: float
    invalid_json_rate: float
    review_rate: float
    mean_latency_ms: float


async def run_eval(ctx: ExtractionAppContext, examples: list[EvalExample]) -> tuple[EvalSummary, list[EvalCaseResult]]:
    case_results = []
    for example in examples:
        record = await run_extraction(ctx, schema_name=example.schema_name, text=example.text)
        case_results.append(
            EvalCaseResult(
                id=example.id,
                schema_name=example.schema_name,
                field_exact_match=field_exact_match(record.extracted_fields, example.reference),
                missing_field_rate=missing_field_rate(record.extracted_fields, example.reference),
                hallucinated_field_rate=hallucinated_field_rate(record.extracted_fields, example.reference),
                invalid_json=record.validation_error is not None,
                needs_review=record.needs_review,
                latency_ms=record.latency_ms,
            )
        )

    n = len(case_results)
    summary = EvalSummary(
        total=n,
        mean_field_exact_match=sum(c.field_exact_match for c in case_results) / n,
        mean_missing_field_rate=sum(c.missing_field_rate for c in case_results) / n,
        mean_hallucinated_field_rate=sum(c.hallucinated_field_rate for c in case_results) / n,
        invalid_json_rate=sum(1 for c in case_results if c.invalid_json) / n,
        review_rate=sum(1 for c in case_results if c.needs_review) / n,
        mean_latency_ms=sum(c.latency_ms for c in case_results) / n,
    )
    return summary, case_results


def summary_to_markdown(summary: EvalSummary, scenario: str) -> str:
    return (
        f"### Scenario: {scenario}\n\n"
        f"- Examples: {summary.total}\n"
        f"- Mean field exact match: {summary.mean_field_exact_match:.2%}\n"
        f"- Mean missing field rate: {summary.mean_missing_field_rate:.2%}\n"
        f"- Mean hallucinated field rate: {summary.mean_hallucinated_field_rate:.2%}\n"
        f"- Invalid JSON rate: {summary.invalid_json_rate:.2%}\n"
        f"- Review rate: {summary.review_rate:.2%}\n"
        f"- Mean latency: {summary.mean_latency_ms:.4f}ms\n"
    )


async def run_lab() -> dict:
    examples = load_dataset(DATASET_PATH)

    with tempfile.TemporaryDirectory(prefix="project01-eval-") as tmp_dir:
        config = AppConfig.model_validate(
            {
                "app": {"data_dir": tmp_dir},
                "models": {
                    "default_chat": "a",
                    "default_extraction": "eval-model",
                    "default_code": "c",
                    "default_embedding": "d",
                },
            }
        )

        perfect_ctx = build_extraction_context(
            config, model_catalog_path=CATALOG_PATH, runtime=DatasetAwareRuntime(examples)
        )
        perfect_summary, _ = await run_eval(perfect_ctx, examples)

        corrupt_ids = frozenset({"inv-004", "tkt-005"})
        imperfect_ctx = build_extraction_context(
            config, model_catalog_path=CATALOG_PATH, runtime=DatasetAwareRuntime(examples, corrupt_ids=corrupt_ids)
        )
        imperfect_summary, _ = await run_eval(imperfect_ctx, examples)

    return {"perfect": perfect_summary, "imperfect": imperfect_summary}


def result_to_markdown(result: dict) -> str:
    return (
        "# Evaluation against the labeled extraction dataset\n\n"
        f"{summary_to_markdown(result['perfect'], 'perfect (proves metrics score a flawless run correctly)')}\n"
        f"{summary_to_markdown(result['imperfect'], 'imperfect (proves metrics catch a real, deliberately broken run)')}\n"
    )


def main(argv: list[str] | None = None) -> int:
    result = asyncio.run(run_lab())
    print(result_to_markdown(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Evaluation command against a labeled dataset (PROPOSAL.md "How success
is measured"). Ingests the real, committed 3-page `multi_page_form.pdf`
fixture through the real composition root, then scores it against the
real golden set (`doc_golden_set.jsonl`) - real routing, real text-layer
extraction, real citation grounding, real latency/memory. The one real
LLM/VLM call stays `FakeRuntime`/`FakeVLM`-backed, scripted to be
per-input-aware (matching Project 1's `DatasetAwareRuntime` and Project
2's `GoldenAwareRuntime` precedent) so the metrics below score a real (if
scripted) answer, not an invented one - real model quality is honest-skip,
deferred to the resourced 32GB Mac.
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT.parent.parent / "packages"))
sys.path.insert(0, str(_PROJECT_ROOT / "app"))
sys.path.insert(0, str(_PROJECT_ROOT / "schemas"))
sys.path.insert(0, str(_PROJECT_ROOT / "prompts"))
sys.path.insert(0, str(_PROJECT_ROOT / "evals"))

from local_ai_core.deployment.config import AppConfig  # noqa: E402
from local_ai_core.multimodal.vlm import FakeVLM  # noqa: E402
from local_ai_core.runtimes.fake import FakeRuntime  # noqa: E402

from doc_eval_metrics import (  # noqa: E402
    answer_contains_expected_page_citation,
    answer_contains_expected_substring,
    bytes_to_mb,
    current_process_rss_bytes,
    field_exact_match,
    route_matches_expected,
    text_layer_char_count_matches,
)
from doc_service import build_doc_context, run_ingest, run_query  # noqa: E402

REPO_ROOT = _PROJECT_ROOT.parent.parent
GOLDEN_SET_PATH = Path(__file__).resolve().parent / "doc_golden_set.jsonl"
FIXTURE_PATH = REPO_ROOT / "datasets" / "multimodal" / "project_04" / "multi_page_form.pdf"
CATALOG_PATH = REPO_ROOT / "models" / "MODEL_CATALOG.md"


@dataclass(frozen=True)
class PageGoldenCase:
    page_id: str
    expected_route: str
    expected_text: str
    reference_fields: dict | None


@dataclass(frozen=True)
class QuestionGoldenCase:
    question_id: str
    question: str
    expected_page_id: str
    must_contain: str


def load_golden_set(path: str | Path) -> tuple[list[PageGoldenCase], list[QuestionGoldenCase]]:
    pages: list[PageGoldenCase] = []
    questions: list[QuestionGoldenCase] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record["type"] == "page":
                pages.append(
                    PageGoldenCase(
                        page_id=record["page_id"],
                        expected_route=record["expected_route"],
                        expected_text=record["expected_text"],
                        reference_fields=record["reference_fields"],
                    )
                )
            else:
                questions.append(
                    QuestionGoldenCase(
                        question_id=record["question_id"],
                        question=record["question"],
                        expected_page_id=record["expected_page_id"],
                        must_contain=record["must_contain"],
                    )
                )
    return pages, questions


class DatasetAwareRuntime(FakeRuntime):
    """Returns the reference extraction JSON for whichever page's text is
    embedded in the extraction prompt, and a scripted, citation-bearing
    answer for whichever question is embedded in a Q&A prompt - one real,
    per-input-aware fake serving both call sites `DocAppContext.base.runtime`
    is shared between (extraction and Q&A both call `ctx.base.runtime`).
    """

    def __init__(self, pages: list[PageGoldenCase], questions: list[QuestionGoldenCase]) -> None:
        super().__init__()
        self._pages = pages
        self._questions = questions

    async def generate(self, request):
        matching_question = next((q for q in self._questions if q.question in request.prompt), None)
        if matching_question is not None:
            text = f"{matching_question.must_contain} [{matching_question.expected_page_id}]."
            self.responses = {request.model: text}
            return await super().generate(request)

        matching_page = next(
            (p for p in self._pages if p.expected_text and p.expected_text in request.prompt), None
        )
        if matching_page is not None and matching_page.reference_fields is not None:
            payload = {**matching_page.reference_fields, "confidence": "high", "evidence": {}}
            text = json.dumps(payload)
        else:
            text = "{}"
        self.responses = {request.model: text}
        return await super().generate(request)


class AdversarialRuntime(DatasetAwareRuntime):
    """Same discipline as Project 1's `corrupt_ids` / Project 3's
    deliberately-failing test: a real, deliberately broken run, proving
    the metrics below catch a real failure rather than only ever seeing
    successes. Every extraction returns wrong field values, and every
    answer cites a page that was never analyzed (an invented citation,
    curriculum's own "citations may be invented" gotcha).
    """

    async def generate(self, request):
        matching_question = next((q for q in self._questions if q.question in request.prompt), None)
        if matching_question is not None:
            text = f"{matching_question.must_contain} [multi_page_form::page99]."
            self.responses = {request.model: text}
            return await FakeRuntime.generate(self, request)

        matching_page = next(
            (p for p in self._pages if p.expected_text and p.expected_text in request.prompt), None
        )
        if matching_page is not None and matching_page.reference_fields is not None:
            # Schema-valid but deliberately wrong on every field (not a
            # type mismatch that would just fail validation and hide
            # behind a different metric - a real, scoreable wrong answer).
            payload = {
                "document_type": "wrong-document-type",
                "applicant_name": "Wrong Name",
                "key_date": "1999-01-01",
                "key_amount": 0.01,
                "notes": None,
                "confidence": "low",
                "evidence": {},
            }
            text = json.dumps(payload)
        else:
            text = "{}"
        self.responses = {request.model: text}
        return await FakeRuntime.generate(self, request)


@dataclass(frozen=True)
class PageEvalResult:
    page_id: str
    route_correct: bool
    text_layer_matches: bool
    field_exact_match: float | None


@dataclass(frozen=True)
class QuestionEvalResult:
    question_id: str
    citation_correct: bool
    citation_verified: bool
    answer_correct: bool
    latency_ms: float


@dataclass(frozen=True)
class DocEvalSummary:
    total_pages: int
    route_accuracy: float
    text_layer_fidelity: float
    mean_field_exact_match: float
    total_questions: int
    citation_correctness_rate: float
    citation_verification_rate: float
    answer_correctness_rate: float
    mean_latency_ms: float
    peak_rss_mb: float


async def run_eval(
    pages: list[PageGoldenCase],
    questions: list[QuestionGoldenCase],
    *,
    runtime_factory=lambda p, q: DatasetAwareRuntime(p, q),
) -> tuple[DocEvalSummary, list[PageEvalResult], list[QuestionEvalResult]]:
    rss_before_mb = bytes_to_mb(current_process_rss_bytes())
    peak_rss_mb = rss_before_mb

    with tempfile.TemporaryDirectory(prefix="project04-eval-") as tmp_dir:
        config = AppConfig.model_validate(
            {
                "app": {"data_dir": tmp_dir},
                "models": {
                    "default_chat": "eval-chat-model",
                    "default_extraction": "eval-extraction-model",
                    "default_code": "c",
                    "default_embedding": "d",
                },
            }
        )
        runtime = runtime_factory(pages, questions)
        ctx = build_doc_context(config, model_catalog_path=CATALOG_PATH, runtime=runtime, vlm=FakeVLM())

        ingestion_result = await run_ingest(ctx, FIXTURE_PATH)
        peak_rss_mb = max(peak_rss_mb, bytes_to_mb(current_process_rss_bytes()))

        analyzed_by_id = {page.page_id: page for page in ingestion_result.pages}
        page_results = []
        for case in pages:
            analyzed = analyzed_by_id[case.page_id]
            page_results.append(
                PageEvalResult(
                    page_id=case.page_id,
                    route_correct=route_matches_expected(analyzed.route, case.expected_route),
                    text_layer_matches=text_layer_char_count_matches(
                        analyzed.extracted_text if analyzed.route == "text_llm" else case.expected_text,
                        case.expected_text,
                    ),
                    field_exact_match=(
                        field_exact_match(analyzed.extracted_fields or {}, case.reference_fields)
                        if case.reference_fields is not None
                        else None
                    ),
                )
            )

        question_results = []
        for case in questions:
            qa_result = await run_query(ctx, "multi_page_form", case.question)
            peak_rss_mb = max(peak_rss_mb, bytes_to_mb(current_process_rss_bytes()))
            citation_correct = answer_contains_expected_page_citation(qa_result.answer, case.expected_page_id)
            citation_verified = any(c.page_id == case.expected_page_id and c.verified for c in qa_result.citations)
            question_results.append(
                QuestionEvalResult(
                    question_id=case.question_id,
                    citation_correct=citation_correct,
                    citation_verified=citation_verified,
                    answer_correct=answer_contains_expected_substring(qa_result.answer, case.must_contain),
                    latency_ms=qa_result.latency_ms,
                )
            )

    n_pages = len(page_results)
    n_questions = len(question_results)
    field_scores = [r.field_exact_match for r in page_results if r.field_exact_match is not None]
    summary = DocEvalSummary(
        total_pages=n_pages,
        route_accuracy=sum(1 for r in page_results if r.route_correct) / n_pages,
        text_layer_fidelity=sum(1 for r in page_results if r.text_layer_matches) / n_pages,
        mean_field_exact_match=(sum(field_scores) / len(field_scores)) if field_scores else 1.0,
        total_questions=n_questions,
        citation_correctness_rate=sum(1 for r in question_results if r.citation_correct) / n_questions,
        citation_verification_rate=sum(1 for r in question_results if r.citation_verified) / n_questions,
        answer_correctness_rate=sum(1 for r in question_results if r.answer_correct) / n_questions,
        mean_latency_ms=sum(r.latency_ms for r in question_results) / n_questions,
        peak_rss_mb=peak_rss_mb,
    )
    return summary, page_results, question_results


def summary_to_markdown(summary: DocEvalSummary, scenario: str) -> str:
    return (
        f"### Scenario: {scenario}\n\n"
        f"- Pages evaluated: {summary.total_pages}\n"
        f"- Route accuracy: {summary.route_accuracy:.2%}\n"
        f"- Text-layer fidelity: {summary.text_layer_fidelity:.2%}\n"
        f"- Mean field exact match: {summary.mean_field_exact_match:.2%}\n"
        f"- Questions evaluated: {summary.total_questions}\n"
        f"- Citation correctness rate: {summary.citation_correctness_rate:.2%}\n"
        f"- Citation verification rate: {summary.citation_verification_rate:.2%}\n"
        f"- Answer correctness rate: {summary.answer_correctness_rate:.2%}\n"
        f"- Mean query latency: {summary.mean_latency_ms:.4f}ms\n"
        f"- Peak RSS: {summary.peak_rss_mb:.1f} MB\n"
    )


async def run_lab() -> dict:
    pages, questions = load_golden_set(GOLDEN_SET_PATH)
    perfect_summary, _, _ = await run_eval(pages, questions)
    adversarial_summary, _, _ = await run_eval(
        pages, questions, runtime_factory=lambda p, q: AdversarialRuntime(p, q)
    )
    return {"perfect": perfect_summary, "adversarial": adversarial_summary}


def result_to_markdown(result: dict) -> str:
    return (
        "# Evaluation against the multimodal document golden set\n\n"
        f"{summary_to_markdown(result['perfect'], 'perfect (proves metrics score a flawless run correctly)')}\n"
        f"{summary_to_markdown(result['adversarial'], 'adversarial (proves metrics catch a real, deliberately broken run)')}\n"
    )


def main(argv: list[str] | None = None) -> int:
    result = asyncio.run(run_lab())
    print(result_to_markdown(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

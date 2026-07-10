"""answer_question — page-citation-verified Q&A (ARCHITECTURE.md "Q&A with
page citations"). Composes `doc_prompts.build_doc_qa_prompt()` (new,
page-id-shaped citation convention) with `citations_are_grounded()`
(Project 2, reused unchanged - id-format-agnostic). Quarantined pages are
never included in the context passed to the model, the same exclusion
`rag_ingestion_service.py` applies to quarantined documents.
"""

from __future__ import annotations

from dataclasses import dataclass

from local_ai_core.runtimes.base import Timer
from local_ai_core.runtimes.types import LLMRequest

from doc_prompts import build_doc_qa_prompt, extract_page_citations
from doc_storage import PageAnalysisRecord
from local_ai_core.evals.citation_verifier import citations_are_grounded


@dataclass(frozen=True)
class PageCitation:
    page_id: str
    page_number: int
    verified: bool


@dataclass(frozen=True)
class DocQaResult:
    answer: str
    citations: list[PageCitation]
    latency_ms: float


def _analyzed_pages(pages: list[PageAnalysisRecord]) -> list[PageAnalysisRecord]:
    return [page for page in pages if page.route != "quarantined"]


async def answer_question(
    pages: list[PageAnalysisRecord], question: str, *, runtime, model: str
) -> DocQaResult:
    analyzed = _analyzed_pages(pages)
    prompt = build_doc_qa_prompt(question, analyzed)

    timer = Timer()
    response = await runtime.generate(LLMRequest(model=model, prompt=prompt))
    latency_ms = timer.elapsed_ms

    analyzed_page_ids = [page.page_id for page in analyzed]
    page_number_by_id = {page.page_id: page.page_number for page in analyzed}

    citations = []
    for page_id in extract_page_citations(response.text):
        verified = citations_are_grounded([page_id], analyzed_page_ids)
        citations.append(
            PageCitation(page_id=page_id, page_number=page_number_by_id.get(page_id, -1), verified=verified)
        )

    return DocQaResult(answer=response.text, citations=citations, latency_ms=latency_ms)

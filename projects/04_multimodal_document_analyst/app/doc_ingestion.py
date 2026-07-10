"""ingest_document — orchestrates the full per-page ingestion pipeline
(ARCHITECTURE.md's high-level diagram): screen -> route -> extract-or-
describe -> persist. The first real wiring of `screen_document_for_ingestion()`
(Module 22) into per-page document ingestion, the same way Project 2's
`rag_ingestion_service.py` first wired it into RAG ingestion.

A quarantined page is not an error (Project 2's framing, reused): the
request succeeds, that page is recorded with a `quarantine_reason`, and -
because it failed screening - is deliberately never rendered to an image
or sent to a model at all, excluding it from extraction/Q&A context.

A VLM-routed page always has `needs_review=True`: unlike the TEXT_LLM
route, no structured schema validates the VLM's output, so nothing has
verified its quality (a real, documented distinction from the `text_llm`
route's confidence scoring).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from local_ai_core.extraction.pipeline import ExtractionPipeline
from local_ai_core.multimodal.pdf_extraction import render_page_to_image
from local_ai_core.multimodal.routing import MultimodalRoute
from local_ai_core.security.rag_ingestion_guard import SourceTrust, screen_document_for_ingestion
from local_ai_rag.loaders.pdf_loader import load_pdf_document

from doc_extraction import extract_page_fields
from doc_routing import decide_route
from doc_schemas import DocumentFieldExtraction
from doc_storage import DocStorage, DocumentRecord, PageAnalysisRecord

DESCRIBE_PROMPT = (
    "Describe the contents of this document page in detail, including any "
    "visible text, signatures, or form fields."
)


class VisionLanguageModel(Protocol):
    async def describe(self, image, prompt: str) -> str: ...


@dataclass(frozen=True)
class IngestionResult:
    doc_id: str
    page_count: int
    pages: list[PageAnalysisRecord]


async def ingest_document(
    pdf_path: str | Path,
    *,
    pipeline: ExtractionPipeline[DocumentFieldExtraction],
    vlm: VisionLanguageModel,
    storage: DocStorage,
    model: str,
    context_window: int,
    min_text_chars: int = 40,
    dpi: int = 150,
) -> IngestionResult:
    documents = load_pdf_document(pdf_path)
    doc_id = Path(pdf_path).stem
    pages: list[PageAnalysisRecord] = []

    for page_number, document in enumerate(documents, start=1):
        screen_decision = screen_document_for_ingestion(document.text, source_trust=SourceTrust.UNTRUSTED)
        if not screen_decision.allowed:
            record = PageAnalysisRecord(
                page_id=document.doc_id,
                doc_id=doc_id,
                page_number=page_number,
                route="quarantined",
                route_reason=screen_decision.reason,
                extracted_text="",
                extracted_fields=None,
                confidence=None,
                needs_review=True,
                quarantine_reason=screen_decision.reason,
            )
            storage.save_page_analysis(record)
            pages.append(record)
            continue

        image = None
        needs_image = len(document.text.strip()) < min_text_chars
        if needs_image:
            image = render_page_to_image(pdf_path, page_number - 1, dpi=dpi)
        routing_decision = decide_route(
            document.text, image=image, context_window=context_window, min_text_chars=min_text_chars
        )

        if routing_decision.route == MultimodalRoute.TEXT_LLM:
            extraction_result = await extract_page_fields(document.text, pipeline=pipeline, model=model)
            record = PageAnalysisRecord(
                page_id=document.doc_id,
                doc_id=doc_id,
                page_number=page_number,
                route=routing_decision.route.value,
                route_reason=routing_decision.reason,
                extracted_text=document.text,
                extracted_fields=extraction_result.fields,
                confidence=extraction_result.confidence,
                needs_review=extraction_result.needs_review,
                quarantine_reason=None,
            )
        else:
            description = await vlm.describe(image, DESCRIBE_PROMPT)
            record = PageAnalysisRecord(
                page_id=document.doc_id,
                doc_id=doc_id,
                page_number=page_number,
                route=routing_decision.route.value,
                route_reason=routing_decision.reason,
                extracted_text=description,
                extracted_fields=None,
                confidence=None,
                needs_review=True,
                quarantine_reason=None,
            )

        storage.save_page_analysis(record)
        pages.append(record)

    storage.save_document(
        DocumentRecord(doc_id=doc_id, source_path=str(pdf_path), page_count=len(documents), status="ingested")
    )
    return IngestionResult(doc_id=doc_id, page_count=len(documents), pages=pages)

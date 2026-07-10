"""DocAppContext — the composition root for this project, extending (not
replacing) Module 23's `AppContext` with a `DocStorage` handle and a
`VisionLanguageModel` (ARCHITECTURE.md "Deployment shape"). Same pattern
Project 1's `extraction_service.py` and Project 2's `rag_service.py`
established for Module 23's composition root. `run_ingest`, `run_extract`,
and `run_query` are the three functions the FastAPI layer calls - it never
reaches past this file into `doc_ingestion`/`doc_extraction`/`doc_qa`
directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from local_ai_core.deployment.app_context import AppContext, build_app_context
from local_ai_core.deployment.config import AppConfig
from local_ai_core.extraction.pipeline import ExtractionPipeline
from local_ai_core.multimodal.vlm import FakeVLM
from local_ai_core.runtimes.base import LLMRuntime

from doc_extraction import extract_page_fields
from doc_ingestion import VisionLanguageModel, ingest_document, IngestionResult
from doc_qa import DocQaResult, answer_question
from doc_schemas import DocumentFieldExtraction
from doc_storage import DocStorage, PageAnalysisRecord


@dataclass
class DocAppContext:
    base: AppContext
    storage: DocStorage
    vlm: VisionLanguageModel


def build_doc_context(
    config: AppConfig,
    *,
    model_catalog_path: str | Path,
    runtime: LLMRuntime | None = None,
    vlm: VisionLanguageModel | None = None,
) -> DocAppContext:
    """`vlm` defaults to `FakeVLM` - this repo's standing honest-skip
    default (Module 18's own pattern); `MlxVisionLanguageModel` is the
    documented "enable for real" path on the resourced Mac, no other code
    change needed since `DocAppContext` takes the VLM via dependency
    injection, same as `AppContext` already does for the runtime.
    """
    base = build_app_context(config, model_catalog_path=model_catalog_path, runtime=runtime)
    doc_dir = base.data_dir.base_dir / "multimodal"
    doc_dir.mkdir(parents=True, exist_ok=True)
    storage = DocStorage(doc_dir / "multimodal.db")
    resolved_vlm = vlm or FakeVLM()
    return DocAppContext(base=base, storage=storage, vlm=resolved_vlm)


async def run_ingest(ctx: DocAppContext, pdf_path: str | Path, *, model: str | None = None) -> IngestionResult:
    resolved_model = model or ctx.base.config.models.default_extraction
    pipeline: ExtractionPipeline = ExtractionPipeline(ctx.base.runtime, DocumentFieldExtraction)
    return await ingest_document(
        pdf_path,
        pipeline=pipeline,
        vlm=ctx.vlm,
        storage=ctx.storage,
        model=resolved_model,
        context_window=ctx.base.config.limits.max_prompt_tokens,
    )


async def run_extract(
    ctx: DocAppContext, doc_id: str, *, page_number: int | None = None, model: str | None = None
) -> list[PageAnalysisRecord]:
    """Re-runs structured extraction for one page (`page_number`) or every
    `text_llm`-routed page in the document. A `vlm`-routed or quarantined
    page has no fields to extract and is silently skipped, not an error -
    the caller asked for extraction, not for a route it didn't choose.
    """
    resolved_model = model or ctx.base.config.models.default_extraction
    pipeline: ExtractionPipeline = ExtractionPipeline(ctx.base.runtime, DocumentFieldExtraction)

    pages = ctx.storage.list_page_analyses(doc_id)
    if page_number is not None:
        pages = [page for page in pages if page.page_number == page_number]

    updated: list[PageAnalysisRecord] = []
    for page in pages:
        if page.route != "text_llm":
            continue
        extraction_result = await extract_page_fields(page.extracted_text, pipeline=pipeline, model=resolved_model)
        updated_page = PageAnalysisRecord(
            page_id=page.page_id,
            doc_id=page.doc_id,
            page_number=page.page_number,
            route=page.route,
            route_reason=page.route_reason,
            extracted_text=page.extracted_text,
            extracted_fields=extraction_result.fields,
            confidence=extraction_result.confidence,
            needs_review=extraction_result.needs_review,
            quarantine_reason=page.quarantine_reason,
        )
        ctx.storage.save_page_analysis(updated_page)
        updated.append(updated_page)

    return updated


async def run_query(ctx: DocAppContext, doc_id: str, question: str, *, model: str | None = None) -> DocQaResult:
    resolved_model = model or ctx.base.config.models.default_chat
    pages = ctx.storage.list_page_analyses(doc_id)
    return await answer_question(pages, question, runtime=ctx.base.runtime, model=resolved_model)

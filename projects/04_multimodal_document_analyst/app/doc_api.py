"""The FastAPI service (ARCHITECTURE.md "API contract"). Same lazy-context
pattern as Module 23's `api_app.py` and Projects 1/2's own `*_api.py`
files (`get_ctx()` builds and caches on `app.state.ctx` on first use, never
at import time).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT.parent.parent / "packages"))
sys.path.insert(0, str(_PROJECT_ROOT / "app"))
sys.path.insert(0, str(_PROJECT_ROOT / "schemas"))
sys.path.insert(0, str(_PROJECT_ROOT / "prompts"))
sys.path.insert(0, str(_PROJECT_ROOT / "evals"))

from fastapi import FastAPI, HTTPException  # noqa: E402

from local_ai_core.deployment.config import AppConfig, load_config  # noqa: E402
from local_ai_core.deployment.health import run_liveness_check, run_readiness_check  # noqa: E402
from local_ai_core.gateway.queue import QueueFullError  # noqa: E402
from local_ai_core.runtimes.errors import RequestTimeout, RuntimeUnavailable  # noqa: E402

from doc_schemas import (  # noqa: E402
    CitationResponse,
    ExtractRequest,
    IngestDocumentRequest,
    PageIngestResult,
    QueryRequest,
    QueryResponse,
)
from doc_service import DocAppContext, build_doc_context, run_extract, run_ingest, run_query  # noqa: E402

REPO_ROOT = _PROJECT_ROOT.parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "app.example.yaml"
DEFAULT_CATALOG_PATH = REPO_ROOT / "models" / "MODEL_CATALOG.md"


def _build_default_context() -> DocAppContext:
    config_path = os.environ.get("APP_CONFIG_PATH", str(DEFAULT_CONFIG_PATH))
    config: AppConfig = load_config(config_path)
    return build_doc_context(config, model_catalog_path=DEFAULT_CATALOG_PATH)


app = FastAPI(title="Multimodal Document Analyst (Project 4)")


def get_ctx() -> DocAppContext:
    if getattr(app.state, "ctx", None) is None:
        app.state.ctx = _build_default_context()
    return app.state.ctx


@app.get("/health")
def health() -> dict:
    result = run_liveness_check()
    return {"status": "ok" if result.passed else "fail", "detail": result.detail}


@app.get("/ready")
def ready() -> dict:
    ctx = get_ctx()
    result = run_readiness_check(ctx.base.data_dir, ctx.base.model_registry)
    if not result.passed:
        raise HTTPException(status_code=503, detail=result.detail)
    return {"status": "ready", "detail": result.detail}


@app.post("/documents", response_model=list[PageIngestResult])
async def ingest(request: IngestDocumentRequest) -> list[PageIngestResult]:
    source_path = Path(request.source_path)
    if not source_path.exists():
        raise HTTPException(status_code=404, detail=f"no file found at {request.source_path!r}")
    if source_path.suffix.lower() != ".pdf":
        raise HTTPException(status_code=422, detail=f"not a PDF file: {request.source_path!r}")

    ctx = get_ctx()
    result = await run_ingest(ctx, source_path)
    return [
        PageIngestResult(page_id=page.page_id, route=page.route, status="ingested")
        for page in result.pages
    ]


@app.get("/documents/{doc_id}")
def get_document(doc_id: str) -> dict:
    ctx = get_ctx()
    record = ctx.storage.get_document(doc_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"no document found with doc_id {doc_id!r}")
    pages = ctx.storage.list_page_analyses(doc_id)
    return {
        "doc_id": record.doc_id,
        "source_path": record.source_path,
        "page_count": record.page_count,
        "status": record.status,
        "ingested_at": record.ingested_at,
        "pages": [
            {
                "page_id": page.page_id,
                "page_number": page.page_number,
                "route": page.route,
                "route_reason": page.route_reason,
                "extracted_text": page.extracted_text,
                "extracted_fields": page.extracted_fields,
                "confidence": page.confidence,
                "needs_review": page.needs_review,
                "quarantine_reason": page.quarantine_reason,
            }
            for page in pages
        ],
    }


@app.post("/documents/{doc_id}/extract")
async def extract(doc_id: str, request: ExtractRequest) -> list[dict]:
    ctx = get_ctx()
    if ctx.storage.get_document(doc_id) is None:
        raise HTTPException(status_code=404, detail=f"no document found with doc_id {doc_id!r}")

    updated = await run_extract(ctx, doc_id, page_number=request.page_number)
    return [
        {
            "page_id": page.page_id,
            "page_number": page.page_number,
            "extracted_fields": page.extracted_fields,
            "confidence": page.confidence,
            "needs_review": page.needs_review,
        }
        for page in updated
    ]


@app.post("/documents/{doc_id}/query", response_model=QueryResponse)
async def query(doc_id: str, request: QueryRequest) -> QueryResponse:
    ctx = get_ctx()
    if ctx.storage.get_document(doc_id) is None:
        raise HTTPException(status_code=404, detail=f"no document found with doc_id {doc_id!r}")

    async def _run():
        return await run_query(ctx, doc_id, request.question)

    try:
        queued_result = await ctx.base.admission_controller.submit(_run)
    except QueueFullError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except RequestTimeout as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except RuntimeUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    result = queued_result.result
    return QueryResponse(
        answer=result.answer,
        citations=[
            CitationResponse(page_id=c.page_id, page_number=c.page_number, verified=c.verified)
            for c in result.citations
        ],
    )

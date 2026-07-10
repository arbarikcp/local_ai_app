"""The FastAPI service (ARCHITECTURE.md "API contract"). Same lazy-context
pattern as Module 23's `api_app.py` and Project 1's `extraction_api.py`
(`get_ctx()` builds and caches on `app.state.ctx` on first use, never at
import time).
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
from local_ai_core.evals.golden_set import load_golden_set  # noqa: E402
from local_ai_core.gateway.queue import QueueFullError  # noqa: E402
from local_ai_core.runtimes.errors import RequestTimeout, RuntimeUnavailable  # noqa: E402
from local_ai_rag.loaders.markdown_loader import Document, load_markdown_file  # noqa: E402
from local_ai_rag.loaders.pdf_loader import load_pdf_document  # noqa: E402

from rag_api_schemas import (  # noqa: E402
    CitationResponse,
    DeleteResponse,
    DocumentIngestRequest,
    DocumentIngestResponseItem,
    DocumentRecordResponse,
    QueryRequest,
    QueryResponse,
    RagEvalRequest,
    RagEvalSummaryResponse,
    TraceResponse,
)
from rag_ingestion_service import delete_document, ingest_document  # noqa: E402
from rag_query_service import answer_question  # noqa: E402
from rag_service import RagAppContext, build_rag_context  # noqa: E402
from rag_text_loader import load_text_file, load_text_string  # noqa: E402
from run_rag_eval import GOLDEN_SET_PATH, run_eval  # noqa: E402

REPO_ROOT = _PROJECT_ROOT.parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "app.example.yaml"
DEFAULT_CATALOG_PATH = REPO_ROOT / "models" / "MODEL_CATALOG.md"


def _build_default_context() -> RagAppContext:
    config_path = os.environ.get("APP_CONFIG_PATH", str(DEFAULT_CONFIG_PATH))
    config: AppConfig = load_config(config_path)
    return build_rag_context(config, model_catalog_path=DEFAULT_CATALOG_PATH)


app = FastAPI(title="Production Local RAG Service (Project 2)")


def get_ctx() -> RagAppContext:
    if getattr(app.state, "ctx", None) is None:
        app.state.ctx = _build_default_context()
    return app.state.ctx


def _load_documents(request: DocumentIngestRequest) -> list[Document]:
    if request.source_type == "markdown":
        if not request.source_path:
            raise HTTPException(status_code=422, detail="source_path is required for source_type=markdown")
        return [load_markdown_file(Path(request.source_path))]
    if request.source_type == "pdf":
        if not request.source_path:
            raise HTTPException(status_code=422, detail="source_path is required for source_type=pdf")
        return load_pdf_document(Path(request.source_path))
    if request.source_type == "text":
        if request.source_path:
            return [load_text_file(Path(request.source_path))]
        if request.text is not None and request.doc_id:
            return [load_text_string(request.doc_id, request.text)]
        raise HTTPException(
            status_code=422, detail="source_type=text requires either source_path, or both text and doc_id"
        )
    raise HTTPException(status_code=422, detail=f"unsupported source_type: {request.source_type!r}")


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


@app.post("/documents", response_model=list[DocumentIngestResponseItem])
async def ingest(request: DocumentIngestRequest) -> list[DocumentIngestResponseItem]:
    ctx = get_ctx()
    documents = _load_documents(request)

    results = []
    for document in documents:
        result = await ingest_document(
            document, embedder=ctx.embedder, store=ctx.store, metadata_store=ctx.metadata_store
        )
        results.append(
            DocumentIngestResponseItem(
                doc_id=result.doc_id,
                status=result.status,
                chunk_count=result.chunk_count,
                quarantine_reason=result.quarantine_reason,
            )
        )
    return results


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    ctx = get_ctx()
    model = ctx.base.config.models.default_chat

    async def _run():
        return await answer_question(
            embedder=ctx.embedder,
            store=ctx.store,
            runtime=ctx.base.runtime,
            metadata_store=ctx.metadata_store,
            question=request.question,
            k=request.k,
            rewrite=request.rewrite,
            metadata_filter=request.metadata_filter,
            model=model,
        )

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
            CitationResponse(
                document_id=c.document_id, chunk_id=c.chunk_id, score=c.score, text_preview=c.text_preview,
                verified=c.verified,
            )
            for c in result.citations
        ],
        trace=TraceResponse(
            retrieved_chunks=result.retrieved_chunks,
            reranked_chunks=result.reranked_chunks,
            context_tokens=result.context_tokens,
            model=result.model,
        ),
    )


@app.get("/documents/{doc_id}", response_model=DocumentRecordResponse)
def get_document(doc_id: str) -> DocumentRecordResponse:
    ctx = get_ctx()
    record = ctx.metadata_store.get_document(doc_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"no document found with doc_id {doc_id!r}")
    return DocumentRecordResponse(
        doc_id=record.doc_id, source_path=record.source_path, title=record.title, status=record.status,
        chunk_count=record.chunk_count, quarantine_reason=record.quarantine_reason, ingested_at=record.ingested_at,
    )


@app.delete("/documents/{doc_id}", response_model=DeleteResponse)
async def remove_document(doc_id: str) -> DeleteResponse:
    ctx = get_ctx()
    record = ctx.metadata_store.get_document(doc_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"no document found with doc_id {doc_id!r}")
    removed = await delete_document(doc_id, store=ctx.store, metadata_store=ctx.metadata_store)
    return DeleteResponse(deleted=True, chunks_removed=removed)


@app.post("/eval/rag", response_model=RagEvalSummaryResponse)
async def run_rag_evaluation(request: RagEvalRequest) -> RagEvalSummaryResponse:
    """Evaluates whatever is actually deployed - the real ingested corpus
    and the real configured runtime - against a labeled golden set. Not a
    fresh, isolated test corpus: an operator must `POST /documents` the
    relevant corpus first, the same way a real evaluation run would only
    be meaningful against the documents actually in production.
    """
    ctx = get_ctx()
    golden_set_path = Path(request.golden_set_path) if request.golden_set_path else GOLDEN_SET_PATH
    cases = load_golden_set(golden_set_path)
    summary, _ = await run_eval(ctx, cases)
    return RagEvalSummaryResponse(
        total=summary.total,
        mean_recall_at_k=summary.mean_recall_at_k,
        mean_precision_at_k=summary.mean_precision_at_k,
        citation_correctness_rate=summary.citation_correctness_rate,
        mean_faithfulness=summary.mean_faithfulness,
        mean_answer_relevance=summary.mean_answer_relevance,
        abstention_accuracy=summary.abstention_accuracy,
        mean_latency_ms=summary.mean_latency_ms,
        peak_rss_mb=summary.peak_rss_mb,
    )

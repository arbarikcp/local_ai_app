"""Query orchestration (ARCHITECTURE.md "Data flow through a query") —
wraps Module 12's `ProductionRagPipeline` (the entire question-side
pipeline, reused unchanged) and adds the one thing curriculum's own
architecture diagram names as a distinct stage but no existing code turns
into a response the caller can act on: citation verification as a
response-gate rather than just an internal check.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from local_ai_core.evals.citation_verifier import citations_are_grounded
from local_ai_core.runtimes.base import LLMRuntime, Timer, with_retries
from local_ai_core.runtimes.errors import RequestTimeout, RuntimeUnavailable
from local_ai_rag.context_packers.budget_packer import estimate_tokens
from local_ai_rag.embeddings.embedder import Embedder
from local_ai_rag.production_pipeline import ProductionRagPipeline
from local_ai_rag.retrievers.acl import ACLPredicate
from local_ai_rag.stores.vector_store import VectorStore

from rag_metadata_store import QueryLogRecord, RagMetadataStore

_TEXT_PREVIEW_CHARS = 200


def _build_metadata_filter_predicate(metadata_filter: dict | None) -> ACLPredicate | None:
    """An exact-match-per-key predicate built from a plain dict - reuses
    the same `ACLPredicate = Callable[[dict], bool]` shape Module 12's
    `clearance_predicate()`/`tenant_predicate()` already establish, just a
    simpler rule (every requested key must match exactly) suited to a
    caller-supplied filter rather than a fixed access-control policy.
    """
    if not metadata_filter:
        return None

    def predicate(metadata: dict) -> bool:
        return all(metadata.get(key) == value for key, value in metadata_filter.items())

    return predicate


@dataclass(frozen=True)
class CitationView:
    document_id: str
    chunk_id: str
    score: float
    text_preview: str
    verified: bool


@dataclass(frozen=True)
class QueryResult:
    query_id: str
    answer: str
    citations: list[CitationView]
    retrieved_chunks: int
    reranked_chunks: int
    retrieved_doc_ids: list[str]
    context_tokens: int
    model: str
    latency_ms: float


async def answer_question(
    *,
    embedder: Embedder,
    store: VectorStore,
    runtime: LLMRuntime,
    metadata_store: RagMetadataStore | None = None,
    question: str,
    k: int = 5,
    rewrite: bool = False,
    metadata_filter: dict | None = None,
    model: str = "fake-model",
) -> QueryResult:
    acl_predicate = _build_metadata_filter_predicate(metadata_filter)
    pipeline = ProductionRagPipeline(embedder, store, runtime, model=model, acl_predicate=acl_predicate)

    async def _run_pipeline():
        return await pipeline.answer(question, rewrite=rewrite, k=k)

    timer = Timer()
    # Retries the whole pipeline call (retrieval + rerank + pack are cheap,
    # deterministic, and side-effect-free) rather than just the internal
    # LLM call, since `ProductionRagPipeline.answer()` is a single opaque
    # async call - the same transport-retry discipline Project 1's
    # extraction_service.py applied, adapted to a pipeline this project
    # doesn't own the internals of.
    answer = await with_retries(_run_pipeline, retryable=(RuntimeUnavailable, RequestTimeout))
    latency_ms = timer.elapsed_ms

    # SearchResult.doc_id holds the chunk_id in this repo's storage
    # convention (VectorStore.add() is always called with chunk.chunk_id as
    # its "doc_id" positional arg - see rag_ingestion_service.py) - the
    # parent document id lives in metadata["doc_id"] instead.
    retrieved_chunk_ids = [chunk.doc_id for chunk in answer.packed_chunks]
    packed_by_chunk_id = {chunk.doc_id: chunk for chunk in answer.packed_chunks}

    citation_views = []
    for citation in answer.citations:
        grounded = citations_are_grounded([citation], retrieved_chunk_ids)
        packed_chunk = packed_by_chunk_id.get(citation)
        document_id = packed_chunk.metadata.get("doc_id", citation.split("::")[0]) if packed_chunk else citation.split("::")[0]
        score = packed_chunk.score if packed_chunk else 0.0
        text = packed_chunk.text if packed_chunk else ""
        citation_views.append(
            CitationView(
                document_id=document_id,
                chunk_id=citation,
                score=score,
                text_preview=text[:_TEXT_PREVIEW_CHARS],
                verified=grounded,
            )
        )

    context_tokens = sum(estimate_tokens(chunk.text) for chunk in answer.packed_chunks)
    # Deduplicated, order-preserving parent-document ids for the chunks
    # that actually reached context packing - what recall@k/precision@k
    # (evals/run_rag_eval.py) compare against a golden case's
    # `expected_source_ids`, since curriculum's own `GoldenCase` docstring
    # defines relevance at document granularity, not chunk granularity.
    retrieved_doc_ids = list(dict.fromkeys(chunk.metadata.get("doc_id", chunk.doc_id) for chunk in answer.packed_chunks))
    query_id = str(uuid.uuid4())

    if metadata_store is not None:
        metadata_store.log_query(
            QueryLogRecord(
                query_id=query_id,
                question=question,
                answer_text=answer.answer_text,
                citation_count=len(answer.citations),
                verified_citation_count=sum(1 for c in citation_views if c.verified),
                latency_ms=latency_ms,
            )
        )

    return QueryResult(
        query_id=query_id,
        answer=answer.answer_text,
        citations=citation_views,
        retrieved_chunks=answer.trace.candidates_retrieved,
        reranked_chunks=answer.trace.candidates_after_rerank,
        retrieved_doc_ids=retrieved_doc_ids,
        context_tokens=context_tokens,
        model=model,
        latency_ms=latency_ms,
    )

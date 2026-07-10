"""Pydantic request/response schemas for the FastAPI layer (ARCHITECTURE.md
"API contract"). Kept separate from `app/rag_api.py` so the response shape
- curriculum's own exact field names (`document_id`, `chunk_id`, `score`,
`text_preview`, plus `trace.retrieved_chunks`/`reranked_chunks`/
`context_tokens`/`model`) - is reviewable independent of the endpoint
wiring, same split Project 1 used for its extraction schemas.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class DocumentIngestRequest(BaseModel):
    source_type: Literal["markdown", "text", "pdf"]
    source_path: str | None = None
    text: str | None = None
    doc_id: str | None = None


class DocumentIngestResponseItem(BaseModel):
    doc_id: str
    status: str
    chunk_count: int
    quarantine_reason: str | None = None


class QueryRequest(BaseModel):
    question: str
    k: int = 5
    rewrite: bool = False
    metadata_filter: dict | None = None


class CitationResponse(BaseModel):
    document_id: str
    chunk_id: str
    score: float
    text_preview: str
    verified: bool


class TraceResponse(BaseModel):
    retrieved_chunks: int
    reranked_chunks: int
    context_tokens: int
    model: str


class QueryResponse(BaseModel):
    answer: str
    citations: list[CitationResponse]
    trace: TraceResponse


class DocumentRecordResponse(BaseModel):
    doc_id: str
    source_path: str | None
    title: str | None
    status: str
    chunk_count: int
    quarantine_reason: str | None = None
    ingested_at: str | None = None


class DeleteResponse(BaseModel):
    deleted: bool
    chunks_removed: int = Field(ge=0)


class RagEvalRequest(BaseModel):
    golden_set_path: str | None = None


class RagEvalSummaryResponse(BaseModel):
    total: int
    mean_recall_at_k: float
    mean_precision_at_k: float
    citation_correctness_rate: float
    mean_faithfulness: float
    mean_answer_relevance: float
    abstention_accuracy: float
    mean_latency_ms: float
    peak_rss_mb: float

"""DocumentFieldExtraction (ARCHITECTURE.md "Document/form schema") plus the
FastAPI request/response shapes for `app/doc_api.py`. Mirrors
`InvoiceExtraction`'s shape (packages/local_ai_core/extraction/schemas.py) -
a required, model-self-reported `confidence`/`evidence` pair (Module 8's own
schema requirement) - generalized for form-shaped documents rather than
invoices specifically.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class DocumentFieldExtraction(BaseModel):
    document_type: str | None = None
    applicant_name: str | None = None
    key_date: str | None = None
    key_amount: float | None = None
    notes: str | None = None
    confidence: Literal["low", "medium", "high"]
    evidence: dict[str, str] = Field(default_factory=dict)


class IngestDocumentRequest(BaseModel):
    source_path: str


class PageIngestResult(BaseModel):
    page_id: str
    route: str
    status: str


class ExtractRequest(BaseModel):
    page_number: int | None = None


class QueryRequest(BaseModel):
    question: str


class CitationResponse(BaseModel):
    page_id: str
    page_number: int
    verified: bool


class QueryResponse(BaseModel):
    answer: str
    citations: list[CitationResponse]

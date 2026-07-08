"""Simplified RAG-adjacent scorers for the Module 3 benchmark suite.

These are intentionally simplified precursors: full context precision/
recall/faithfulness scoring with a judge model, plus the judge-model-problem
discussion, belongs to Module 13 (RAG v3: evaluation, citations, and
guardrails). Module 3 only needs set-based retrieval metrics and a citation
sanity check to benchmark a model's "grounded answer from provided context"
task.
"""

from __future__ import annotations

import re


def context_precision(retrieved_doc_ids: list[str], relevant_doc_ids: list[str]) -> float:
    """Fraction of retrieved docs that were actually relevant."""
    if not retrieved_doc_ids:
        return 0.0
    relevant_set = set(relevant_doc_ids)
    hits = sum(1 for doc_id in retrieved_doc_ids if doc_id in relevant_set)
    return hits / len(retrieved_doc_ids)


def context_recall(retrieved_doc_ids: list[str], relevant_doc_ids: list[str]) -> float:
    """Fraction of relevant docs that were actually retrieved."""
    if not relevant_doc_ids:
        return 0.0
    retrieved_set = set(retrieved_doc_ids)
    hits = sum(1 for doc_id in relevant_doc_ids if doc_id in retrieved_set)
    return hits / len(relevant_doc_ids)


_CITATION_RE = re.compile(r"\[(doc\d+|[A-Za-z0-9_.-]+)\]")


def extract_citations(answer: str) -> list[str]:
    """Extract bracketed citation ids like ``[doc1]`` from an answer string."""
    return _CITATION_RE.findall(answer)


def citation_validity(answer: str, available_doc_ids: list[str]) -> float:
    """Fraction of citations in the answer that reference a real, available doc id.

    Returns 1.0 (vacuously valid) if the answer makes no citations at all —
    callers that require citations should check ``extract_citations`` is
    non-empty separately.
    """
    citations = extract_citations(answer)
    if not citations:
        return 1.0
    available = set(available_doc_ids)
    valid = sum(1 for c in citations if c in available)
    return valid / len(citations)


def answer_is_grounded_refusal(answer: str, refusal_phrase: str) -> bool:
    """True if the model correctly declined to answer from insufficient context.

    Used for the negative-case rows in rag.jsonl where the golden expectation
    is that the model says it doesn't know rather than fabricating an answer.
    """
    return refusal_phrase.strip().lower() in answer.strip().lower()

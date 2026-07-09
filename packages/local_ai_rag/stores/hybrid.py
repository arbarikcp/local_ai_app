"""Hybrid search — combines keyword and vector retrieval (theory doc
"Hybrid search"). Pure vector top-k can miss a document that shares an
exact term with the query (an order number, a product code, an acronym)
but whose overall embedding isn't the closest match; keyword search alone
misses paraphrases and synonyms. Combining both catches more than either
alone.

Keyword scoring here is a simple term-overlap score (documentation-oriented
"real, if crude" signal in the same spirit as Module 9's `FakeEmbedder`,
not a full BM25 implementation - BM25 adds term-frequency saturation and
inverse document frequency weighting this module doesn't implement).

The two rankings are combined with Reciprocal Rank Fusion (RRF), not a
weighted sum of scores - cosine similarity (bounded [-1, 1]) and a raw
term-overlap ratio aren't on comparable scales, but rank position always is.
"""

from __future__ import annotations

import re

import numpy as np

from local_ai_rag.embeddings.embedder import SearchResult
from local_ai_rag.stores.vector_store import VectorStore

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def keyword_score(query: str, text: str) -> float:
    """Fraction of the document's words that also appear in the query -
    crude, but enough to reward exact-term matches vector similarity can
    underweight.
    """
    query_terms = set(_tokenize(query))
    doc_terms = _tokenize(text)
    if not query_terms or not doc_terms:
        return 0.0
    hits = sum(1 for term in doc_terms if term in query_terms)
    return hits / len(doc_terms)


def keyword_search(query: str, documents: dict[str, str], k: int = 5) -> list[tuple[str, float]]:
    scored = [(doc_id, keyword_score(query, text)) for doc_id, text in documents.items()]
    scored = [(doc_id, score) for doc_id, score in scored if score > 0]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:k]


def reciprocal_rank_fusion(ranked_id_lists: list[list[str]], k_constant: int = 60) -> list[tuple[str, float]]:
    """Standard RRF: score(doc) = sum over rankings of 1 / (k_constant + rank).
    A document ranked highly in multiple lists outscores one ranked highly
    in only one - the entire point of combining two independent signals.
    """
    scores: dict[str, float] = {}
    for ranked_ids in ranked_id_lists:
        for rank, doc_id in enumerate(ranked_ids, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k_constant + rank)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)


async def hybrid_search(
    store: VectorStore,
    documents: dict[str, str],
    query: str,
    query_embedding: np.ndarray,
    k: int = 5,
    fetch_k: int = 20,
) -> list[SearchResult]:
    """`documents` is the full doc_id -> text corpus, needed because
    keyword search must be able to find a document even when it didn't
    make the vector search's top `fetch_k` - the exact failure mode hybrid
    search exists to fix.
    """
    vector_results = await store.search(query_embedding, k=fetch_k)
    vector_ranked_ids = [r.doc_id for r in vector_results]
    keyword_ranked_ids = [doc_id for doc_id, _score in keyword_search(query, documents, k=fetch_k)]

    fused = reciprocal_rank_fusion([vector_ranked_ids, keyword_ranked_ids])
    text_by_id = {r.doc_id: (r.text, r.metadata) for r in vector_results}

    results = []
    for doc_id, score in fused[:k]:
        text, metadata = text_by_id.get(doc_id, (documents.get(doc_id, ""), {}))
        results.append(SearchResult(doc_id=doc_id, score=score, text=text, metadata=metadata))
    return results

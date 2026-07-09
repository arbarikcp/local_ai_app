"""Embedding evaluation: recall@k, precision@k, MRR, nDCG@k, latency, and
embedding throughput (theory doc "Embedding evaluation" - the exact metric
list the curriculum names).

recall_at_k/precision_at_k/reciprocal_rank/ndcg_at_k moved to
`local_ai_core.evals.retrieval_metrics` in Module 13 (generic,
reference-based metrics with no embedding-specific coupling belong in
shared eval infra, not duplicated per task package) - re-exported here so
existing imports (`from local_ai_rag.embeddings.eval import recall_at_k`)
keep working unchanged.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from local_ai_core.evals.retrieval_metrics import (
    ndcg_at_k,
    precision_at_k,
    reciprocal_rank,
    recall_at_k,
)

from .embedder import Embedder, NumpyEmbeddingIndex

__all__ = [
    "EmbeddingEvalCase",
    "EvalCaseResult",
    "EvalSummary",
    "evaluate_embedder",
    "measure_embedding_throughput",
    "ndcg_at_k",
    "precision_at_k",
    "reciprocal_rank",
    "recall_at_k",
]


@dataclass(frozen=True)
class EmbeddingEvalCase:
    """The curriculum's own golden test-set shape."""

    query: str
    positive_doc_ids: list[str]
    negative_doc_ids: list[str] = field(default_factory=list)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


@dataclass(frozen=True)
class EvalCaseResult:
    query: str
    recall_at_k: float
    precision_at_k: float
    reciprocal_rank: float
    ndcg_at_k: float
    latency_seconds: float


@dataclass(frozen=True)
class EvalSummary:
    case_results: list[EvalCaseResult]

    @property
    def mean_recall_at_k(self) -> float:
        return _mean([r.recall_at_k for r in self.case_results])

    @property
    def mean_precision_at_k(self) -> float:
        return _mean([r.precision_at_k for r in self.case_results])

    @property
    def mrr(self) -> float:
        return _mean([r.reciprocal_rank for r in self.case_results])

    @property
    def mean_ndcg_at_k(self) -> float:
        return _mean([r.ndcg_at_k for r in self.case_results])

    @property
    def mean_query_latency_seconds(self) -> float:
        return _mean([r.latency_seconds for r in self.case_results])


async def evaluate_embedder(
    embedder: Embedder, index: NumpyEmbeddingIndex, eval_cases: list[EmbeddingEvalCase], k: int = 5
) -> EvalSummary:
    case_results = []
    for case in eval_cases:
        relevant_ids = set(case.positive_doc_ids)
        start = time.perf_counter()
        query_vector = await embedder.embed_query(case.query)
        results = index.search(query_vector, k=k)
        latency = time.perf_counter() - start
        retrieved_ids = [r.doc_id for r in results]
        case_results.append(
            EvalCaseResult(
                query=case.query,
                recall_at_k=recall_at_k(retrieved_ids, relevant_ids, k),
                precision_at_k=precision_at_k(retrieved_ids, relevant_ids, k),
                reciprocal_rank=reciprocal_rank(retrieved_ids, relevant_ids),
                ndcg_at_k=ndcg_at_k(retrieved_ids, relevant_ids, k),
                latency_seconds=latency,
            )
        )
    return EvalSummary(case_results=case_results)


async def measure_embedding_throughput(embedder: Embedder, texts: list[str]) -> float:
    """Docs/sec for embed_documents() - a separate measurement from query
    latency, since document embedding is typically batched during
    ingestion, not part of the interactive query path.
    """
    if not texts:
        return 0.0
    start = time.perf_counter()
    await embedder.embed_documents(texts)
    elapsed = time.perf_counter() - start
    return len(texts) / elapsed if elapsed > 0 else float("inf")

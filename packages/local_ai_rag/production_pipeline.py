"""ProductionRagPipeline - the query-side half of the theory doc's
production pipeline diagram, wired from every component built this
module: rewrite (optional) -> ACL filter -> retrieve -> rerank -> pack
context -> generate -> validate citations -> log trace.

"classify query" (the diagram's first stage) is not implemented - it only
earns its keep when different query types route to genuinely different
retrieval strategies, and this module has one corpus and one retrieval
strategy to route between; Module 13's evaluation work is a more natural
home for that decision.
"""

from __future__ import annotations

from dataclasses import dataclass

from local_ai_core.runtimes.base import LLMRuntime
from local_ai_core.runtimes.types import LLMRequest
from local_ai_rag.context_packers.budget_packer import ContextBudget, order_for_generation, pack_context
from local_ai_rag.context_packers.citation_packer import (
    build_rag_prompt,
    extract_citations,
    summarize_source_citations,
)
from local_ai_rag.embeddings.embedder import Embedder, SearchResult
from local_ai_rag.rerankers.heuristic_reranker import HeuristicReranker
from local_ai_rag.retrievers.acl import ACLPredicate
from local_ai_rag.retrievers.naive_retriever import NaiveRetriever
from local_ai_rag.retrievers.query_expansion import rewrite_query
from local_ai_rag.stores.vector_store import VectorStore

_DEFAULT_BUDGET = ContextBudget(
    max_context_tokens=2000, reserved_for_system=200, reserved_for_question=100, reserved_for_answer=400
)


@dataclass(frozen=True)
class TraceLog:
    question: str
    rewritten_question: str | None
    candidates_retrieved: int
    candidates_after_acl: int
    candidates_after_rerank: int
    chunks_packed: int


@dataclass(frozen=True)
class ProductionRagAnswer:
    question: str
    answer_text: str
    packed_chunks: list[SearchResult]
    citations: list[str]
    source_citations: list[str]
    trace: TraceLog

    @property
    def citations_are_grounded(self) -> bool:
        packed_ids = {c.doc_id for c in self.packed_chunks}
        return all(citation in packed_ids for citation in self.citations)


class ProductionRagPipeline:
    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        runtime: LLMRuntime,
        *,
        model: str = "fake-model",
        reranker: HeuristicReranker | None = None,
        acl_predicate: ACLPredicate | None = None,
        context_budget: ContextBudget | None = None,
        fetch_k: int = 20,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._runtime = runtime
        self._model = model
        self._retriever = NaiveRetriever(embedder, store)
        self._reranker = reranker or HeuristicReranker()
        self._acl_predicate = acl_predicate
        self._context_budget = context_budget or _DEFAULT_BUDGET
        self._fetch_k = fetch_k

    async def answer(self, question: str, *, rewrite: bool = False, k: int = 5) -> ProductionRagAnswer:
        rewritten_question: str | None = None
        query_for_retrieval = question
        if rewrite:
            rewritten_question = await rewrite_query(question, self._runtime, self._model)
            query_for_retrieval = rewritten_question

        candidates = await self._retriever.retrieve(query_for_retrieval, k=self._fetch_k)

        allowed = (
            [c for c in candidates if self._acl_predicate(c.metadata)]
            if self._acl_predicate is not None
            else candidates
        )

        reranked = self._reranker.rerank(query_for_retrieval, allowed, k=k)

        packed = pack_context(reranked, self._context_budget)
        ordered = order_for_generation(packed)

        prompt = build_rag_prompt(question, ordered)
        response = await self._runtime.generate(LLMRequest(model=self._model, prompt=prompt))

        citations = extract_citations(response.text)

        trace = TraceLog(
            question=question,
            rewritten_question=rewritten_question,
            candidates_retrieved=len(candidates),
            candidates_after_acl=len(allowed),
            candidates_after_rerank=len(reranked),
            chunks_packed=len(packed),
        )

        return ProductionRagAnswer(
            question=question,
            answer_text=response.text,
            packed_chunks=ordered,
            citations=citations,
            source_citations=summarize_source_citations(citations),
            trace=trace,
        )

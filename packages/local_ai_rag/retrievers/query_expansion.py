"""Query rewriting, multi-query retrieval, and HyDE (theory doc §7-9) - all
three transform the query *before* retrieval using one LLM call, wired
against Module 6's `LLMRuntime` protocol (`FakeRuntime` here, a real
adapter unchanged on the resourced Mac).
"""

from __future__ import annotations

from local_ai_core.runtimes.base import LLMRuntime
from local_ai_core.runtimes.types import LLMRequest
from local_ai_rag.embeddings.embedder import Embedder, SearchResult
from local_ai_rag.retrievers.naive_retriever import NaiveRetriever
from local_ai_rag.stores.hybrid import reciprocal_rank_fusion
from local_ai_rag.stores.vector_store import VectorStore

REWRITE_PROMPT_TEMPLATE = """Rewrite the following question to be more specific and \
searchable, using terminology likely to appear in the source documents. Reply with only \
the rewritten question, nothing else.

Question: {question}

Rewritten question:"""

MULTI_QUERY_PROMPT_TEMPLATE = """Generate {n} different ways to ask the following question, \
each capturing a different phrasing or angle. Reply with exactly {n} lines, one question per \
line, nothing else.

Question: {question}

Alternate phrasings:"""

HYDE_PROMPT_TEMPLATE = """Write a short, plausible passage that would answer the following \
question, as if it were an excerpt from a real document. Do not mention that this is \
hypothetical.

Question: {question}

Passage:"""


async def rewrite_query(question: str, runtime: LLMRuntime, model: str) -> str:
    request = LLMRequest(model=model, prompt=REWRITE_PROMPT_TEMPLATE.format(question=question))
    response = await runtime.generate(request)
    return response.text.strip()


async def rewrite_and_retrieve(
    question: str, retriever: NaiveRetriever, runtime: LLMRuntime, model: str, k: int = 5
) -> list[SearchResult]:
    rewritten = await rewrite_query(question, runtime, model)
    return await retriever.retrieve(rewritten, k=k)


async def generate_query_variants(question: str, runtime: LLMRuntime, model: str, n: int = 3) -> list[str]:
    request = LLMRequest(model=model, prompt=MULTI_QUERY_PROMPT_TEMPLATE.format(question=question, n=n))
    response = await runtime.generate(request)
    variants = [line.strip() for line in response.text.splitlines() if line.strip()]
    return variants[:n] if variants else [question]


async def multi_query_retrieve(
    question: str, retriever: NaiveRetriever, runtime: LLMRuntime, model: str, n_queries: int = 3, k: int = 5
) -> list[SearchResult]:
    """Each query variant is retrieved separately; results are fused with
    Reciprocal Rank Fusion (Module 10's `hybrid.py`) rather than averaging
    raw scores - the same reasoning hybrid search uses RRF: rank position
    is comparable across independently-scored result sets, raw scores
    aren't guaranteed to be.
    """
    variants = await generate_query_variants(question, runtime, model, n=n_queries)
    all_variants = [question, *variants]

    ranked_id_lists: list[list[str]] = []
    result_by_id: dict[str, SearchResult] = {}
    for variant in all_variants:
        results = await retriever.retrieve(variant, k=k)
        ranked_id_lists.append([r.doc_id for r in results])
        for r in results:
            result_by_id.setdefault(r.doc_id, r)

    fused = reciprocal_rank_fusion(ranked_id_lists)
    return [result_by_id[doc_id] for doc_id, _score in fused[:k]]


async def hyde_retrieve(
    question: str, embedder: Embedder, store: VectorStore, runtime: LLMRuntime, model: str, k: int = 5
) -> list[SearchResult]:
    """Hypothetical Document Embeddings: embed a hypothetical *answer* to
    the question, not the question itself - a plausible answer passage is
    often closer, in embedding space, to a real answer chunk than the
    question's own phrasing is.
    """
    request = LLMRequest(model=model, prompt=HYDE_PROMPT_TEMPLATE.format(question=question))
    response = await runtime.generate(request)
    hypothetical_embedding = await embedder.embed_query(response.text.strip())
    return await store.search(hypothetical_embedding, k=k)

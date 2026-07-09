"""ACL-aware retrieval (theory doc §14) - real access-control enforcement
at the retrieval layer, not left to the prompt or the generator to "not
mention" restricted content. `VectorStore.metadata_filter` (Module 10) is
exact-match only; ACL rules like `security_level <= user_clearance` need a
predicate, not an equality check, so filtering happens client-side over
an over-fetched candidate set.
"""

from __future__ import annotations

from typing import Any, Callable, Protocol

from local_ai_rag.embeddings.embedder import SearchResult

ACLPredicate = Callable[[dict[str, Any]], bool]


class Retriever(Protocol):
    async def retrieve(self, query: str, k: int = 5) -> list[SearchResult]: ...


class AclAwareRetriever:
    def __init__(self, retriever: Retriever, acl_predicate: ACLPredicate, *, fetch_multiplier: int = 4) -> None:
        """`fetch_multiplier` over-fetches from the wrapped retriever so
        that documents removed by the ACL check don't silently shrink the
        effective top-k below what the caller asked for.
        """
        self._retriever = retriever
        self._acl_predicate = acl_predicate
        self._fetch_multiplier = fetch_multiplier

    async def retrieve(self, query: str, k: int = 5) -> list[SearchResult]:
        candidates = await self._retriever.retrieve(query, k=k * self._fetch_multiplier)
        allowed = [r for r in candidates if self._acl_predicate(r.metadata)]
        return allowed[:k]


def clearance_predicate(user_clearance: int, *, security_level_key: str = "security_level") -> ACLPredicate:
    """A document with no security_level set is treated as public
    (clearance 0) - the safe default for a caller that forgot to tag a
    document, not a permissive accident.
    """

    def predicate(metadata: dict[str, Any]) -> bool:
        document_level = metadata.get(security_level_key, 0)
        return document_level <= user_clearance

    return predicate


def tenant_predicate(tenant_id: str, *, tenant_id_key: str = "tenant_id") -> ACLPredicate:
    def predicate(metadata: dict[str, Any]) -> bool:
        return metadata.get(tenant_id_key) == tenant_id

    return predicate

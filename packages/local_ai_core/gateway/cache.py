"""Response, semantic, and embedding caches (theory doc §7-11).

Cache keys must include model, quantization, prompt version, tool version,
and safety policy version when those affect output (Gotcha, verbatim from
the bible) - response_cache_key() takes these as required parameters, not
optional extras, so omitting one is a visible choice, not an accident.

Semantic caching can return confidently wrong answers for near-but-not-
equivalent questions - SemanticCache defaults to a conservative (high)
similarity threshold and always returns the matched score alongside a hit,
so a caller can audit borderline matches instead of trusting them blindly.
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np

T = TypeVar("T")


def response_cache_key(
    model: str,
    rendered_prompt: str,
    params: dict[str, object],
    prompt_version: str,
    *,
    quantization: str | None = None,
    tool_version: str | None = None,
    schema_version: str | None = None,
    safety_policy_version: str | None = None,
) -> str:
    """Stable hash key for exact-match response caching.

    The optional version fields default to None (meaning "not applicable to
    this request"), but are always included in the hashed payload - a
    caller that DOES have a quantization/tool/schema/safety-policy version
    that affects output must pass it, or risk a stale cache hit across a
    version change (theory doc §11).
    """
    payload = {
        "model": model,
        "prompt": rendered_prompt,
        "params": params,
        "prompt_version": prompt_version,
        "quantization": quantization,
        "tool_version": tool_version,
        "schema_version": schema_version,
        "safety_policy_version": safety_policy_version,
    }
    serialized = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()


class ResponseCache(Generic[T]):
    """Exact-match cache, keyed by response_cache_key(). LRU eviction."""

    def __init__(self, max_entries: int = 1000) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        self.max_entries = max_entries
        self._store: OrderedDict[str, T] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> T | None:
        if key not in self._store:
            self.misses += 1
            return None
        self.hits += 1
        self._store.move_to_end(key)
        return self._store[key]

    def put(self, key: str, value: T) -> None:
        if key in self._store:
            self._store.move_to_end(key)
        elif len(self._store) >= self.max_entries:
            self._store.popitem(last=False)  # evict least-recently-used
        self._store[key] = value

    def __len__(self) -> int:
        return len(self._store)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Minimal local cosine similarity for the semantic cache.

    Not the canonical embeddings implementation - that's Module 9's job
    (packages/local_ai_rag/embeddings/). This is a small, self-contained
    utility scoped to this module's caching need.
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


@dataclass(frozen=True)
class SemanticCacheEntry(Generic[T]):
    embedding: np.ndarray
    response: T
    original_query: str


@dataclass(frozen=True)
class SemanticCacheHit(Generic[T]):
    response: T
    similarity: float
    matched_query: str


class SemanticCache(Generic[T]):
    """Caches by embedding similarity, not exact match.

    Defaults to a conservative (high) similarity_threshold - lower it only
    after auditing false-hit behavior on your own query distribution
    (theory doc Gotcha).
    """

    def __init__(self, similarity_threshold: float = 0.95, max_entries: int = 500) -> None:
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between 0.0 and 1.0")
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        self.similarity_threshold = similarity_threshold
        self.max_entries = max_entries
        self._entries: list[SemanticCacheEntry[T]] = []
        self.hits = 0
        self.misses = 0

    def get(self, query_embedding: np.ndarray) -> SemanticCacheHit[T] | None:
        best_score = -1.0
        best_entry: SemanticCacheEntry[T] | None = None
        for entry in self._entries:
            score = _cosine_similarity(query_embedding, entry.embedding)
            if score > best_score:
                best_score = score
                best_entry = entry

        if best_entry is not None and best_score >= self.similarity_threshold:
            self.hits += 1
            return SemanticCacheHit(response=best_entry.response, similarity=best_score, matched_query=best_entry.original_query)
        self.misses += 1
        return None

    def put(self, query_embedding: np.ndarray, response: T, original_query: str = "") -> None:
        if len(self._entries) >= self.max_entries:
            self._entries.pop(0)  # FIFO eviction - simplest policy, good enough at this scale
        self._entries.append(
            SemanticCacheEntry(embedding=query_embedding, response=response, original_query=original_query)
        )

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


def embedding_cache_key(text: str, embedding_model: str, normalization_version: str = "v1") -> str:
    """Stable hash key for embedding caching.

    Must be invalidated (a new key) on embedding-model change (theory doc
    §7-10 table) - embedding_model is part of the key precisely so a model
    swap can never silently reuse another model's vectors.
    """
    payload = f"{embedding_model}:{normalization_version}:{text}"
    return hashlib.sha256(payload.encode()).hexdigest()


class EmbeddingCache:
    """Caches embedding vectors, keyed by embedding_cache_key(). LRU eviction.

    Saves re-embedding during ingestion (Lab 7) - ready for Module 9-11's
    ingestion pipeline to use directly.
    """

    def __init__(self, max_entries: int = 10_000) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        self.max_entries = max_entries
        self._store: OrderedDict[str, np.ndarray] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, text: str, embedding_model: str, normalization_version: str = "v1") -> np.ndarray | None:
        key = embedding_cache_key(text, embedding_model, normalization_version)
        if key not in self._store:
            self.misses += 1
            return None
        self.hits += 1
        self._store.move_to_end(key)
        return self._store[key]

    def put(
        self, text: str, embedding_model: str, embedding: np.ndarray, normalization_version: str = "v1"
    ) -> None:
        key = embedding_cache_key(text, embedding_model, normalization_version)
        if key in self._store:
            self._store.move_to_end(key)
        elif len(self._store) >= self.max_entries:
            self._store.popitem(last=False)
        self._store[key] = embedding

    def __len__(self) -> int:
        return len(self._store)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

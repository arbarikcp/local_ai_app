"""FakeEmbedder — a deterministic, non-neural embedder for tests and
infrastructure demos (Module 6's FakeRuntime pattern, applied to embeddings).

Uses bag-of-words feature hashing (the "hashing trick"): each word hashes
into a fixed-size vector, word vectors are summed and normalized. This is a
real, legitimate simple embedding technique - texts sharing more words
genuinely get more similar vectors - but it is NOT a neural embedding model
and captures none of the semantic generalization a real model would
(synonyms get no special treatment). Labeled honestly as a fake for tests
and demos, never a baseline anyone should actually deploy.
"""

from __future__ import annotations

import hashlib

import numpy as np

from .embedder import normalize


class FakeEmbedder:
    def __init__(self, dimensions: int = 32) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self._dimensions = dimensions
        self.embed_documents_call_count = 0
        self.embed_query_call_count = 0

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def _hash_embed(self, text: str) -> np.ndarray:
        vector = np.zeros(self._dimensions)
        for word in text.lower().split():
            digest = int(hashlib.sha256(word.encode()).hexdigest(), 16)
            index = digest % self._dimensions
            sign = 1.0 if (digest // self._dimensions) % 2 == 0 else -1.0
            vector[index] += sign
        return normalize(vector)

    async def embed_documents(self, texts: list[str]) -> list[np.ndarray]:
        self.embed_documents_call_count += 1
        return [self._hash_embed(text) for text in texts]

    async def embed_query(self, text: str) -> np.ndarray:
        self.embed_query_call_count += 1
        return self._hash_embed(text)

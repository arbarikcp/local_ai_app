"""SentenceTransformersEmbedder — Embedder adapter for sentence-transformers
(theory doc "Embedding serving reality": many strong embedders, especially
BGE/GTE/ModernBERT-style models, are best run this way rather than through
a generator's own runtime).

Lazy-imports sentence_transformers and injects the model-loading/encoding
calls via constructor - the same dependency-injection principle as Module
6's MLXRuntime - so tests substitute fakes without the package installed or
a model downloaded (this repo's machine constraint: no model runtime or
weights on this machine at all). Synchronous calls run via
asyncio.to_thread, same reasoning as MLXRuntime: blocking calls inside an
async server can serialize requests.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable

import numpy as np

LoadFn = Callable[[str], Any]
EncodeFn = Callable[[Any, list[str]], Any]


def _real_load(model_name: str) -> Any:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def _real_encode(model: Any, texts: list[str]) -> Any:
    return model.encode(texts)


class SentenceTransformersEmbedder:
    def __init__(
        self,
        model_name: str,
        *,
        load_fn: LoadFn = _real_load,
        encode_fn: EncodeFn = _real_encode,
        query_prefix: str = "",
        document_prefix: str = "",
    ) -> None:
        self.model_name = model_name
        self._load_fn = load_fn
        self._encode_fn = encode_fn
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self._model: Any | None = None
        self._dimensions: int | None = None

    async def _get_model(self) -> Any:
        if self._model is None:
            self._model = await asyncio.to_thread(self._load_fn, self.model_name)
        return self._model

    async def embed_documents(self, texts: list[str]) -> list[np.ndarray]:
        model = await self._get_model()
        prefixed = [f"{self.document_prefix}{text}" for text in texts]
        raw = await asyncio.to_thread(self._encode_fn, model, prefixed)
        vectors = [np.asarray(v, dtype=float) for v in raw]
        if vectors and self._dimensions is None:
            self._dimensions = len(vectors[0])
        return vectors

    async def embed_query(self, text: str) -> np.ndarray:
        model = await self._get_model()
        raw = await asyncio.to_thread(self._encode_fn, model, [f"{self.query_prefix}{text}"])
        vector = np.asarray(raw[0], dtype=float)
        if self._dimensions is None:
            self._dimensions = len(vector)
        return vector

    @property
    def dimensions(self) -> int:
        if self._dimensions is None:
            raise RuntimeError(
                "dimensions unknown until at least one embed call has been made, "
                "since sentence-transformers doesn't expose it before loading the model"
            )
        return self._dimensions

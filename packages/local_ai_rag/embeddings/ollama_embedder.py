"""OllamaEmbedder — Embedder adapter for Ollama's native /api/embeddings
endpoint (theory doc "Embedding serving reality": convenient, benchmark
before trusting against sentence-transformers).

Reuses Module 6's LLMError taxonomy (local_ai_rag depending on
local_ai_core is the expected direction - the reverse, and the
packages-depending-on-scripts direction Module 8 flagged, are not).
Tested via httpx.MockTransport, same pattern as Module 6's OllamaRuntime -
never touches a real server here.
"""

from __future__ import annotations

import httpx
import numpy as np

from local_ai_core.runtimes.errors import InvalidModelResponse, LLMError, RequestTimeout, RuntimeUnavailable

DEFAULT_BASE_URL = "http://localhost:11434"


def map_httpx_error(exc: httpx.HTTPError) -> LLMError:
    if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout)):
        return RuntimeUnavailable(f"Could not connect to Ollama: {exc}", cause=exc)
    if isinstance(exc, httpx.TimeoutException):
        return RequestTimeout(f"Request to Ollama timed out: {exc}", cause=exc)
    return RuntimeUnavailable(f"Could not reach Ollama: {exc}", cause=exc)


class OllamaEmbedder:
    def __init__(
        self,
        model: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 60.0,
        client: httpx.AsyncClient | None = None,
        dimensions: int | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self._client = client or httpx.AsyncClient(timeout=timeout)
        self._owns_client = client is None
        self._dimensions = dimensions  # discovered from the first real response if not given

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def _embed_one(self, text: str) -> np.ndarray:
        try:
            resp = await self._client.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": text},
                timeout=self.timeout,
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise map_httpx_error(exc) from exc

        data = resp.json()
        embedding = data.get("embedding")
        if not isinstance(embedding, list) or not embedding:
            raise InvalidModelResponse(f"Ollama /api/embeddings returned an unexpected shape: {data!r}")

        vector = np.array(embedding, dtype=float)
        if self._dimensions is None:
            self._dimensions = len(vector)
        return vector

    async def embed_documents(self, texts: list[str]) -> list[np.ndarray]:
        return [await self._embed_one(text) for text in texts]

    async def embed_query(self, text: str) -> np.ndarray:
        return await self._embed_one(text)

    @property
    def dimensions(self) -> int:
        if self._dimensions is None:
            raise RuntimeError(
                "dimensions unknown until at least one embed call has been made, "
                "or pass dimensions= explicitly at construction"
            )
        return self._dimensions

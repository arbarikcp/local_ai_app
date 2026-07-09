import httpx
import pytest

from local_ai_core.runtimes.errors import InvalidModelResponse, RequestTimeout, RuntimeUnavailable
from local_ai_rag.embeddings.ollama_embedder import OllamaEmbedder, map_httpx_error


class TestMapHttpxError:
    def test_connect_error_maps_to_runtime_unavailable(self):
        assert isinstance(map_httpx_error(httpx.ConnectError("refused")), RuntimeUnavailable)

    def test_read_timeout_maps_to_request_timeout(self):
        assert isinstance(map_httpx_error(httpx.ReadTimeout("timed out")), RequestTimeout)


def _client_with_handler(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


class TestOllamaEmbedderEmbedQuery:
    async def test_returns_a_numpy_array_from_the_response(self):
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/api/embeddings"
            return httpx.Response(200, json={"embedding": [0.1, 0.2, 0.3]})

        embedder = OllamaEmbedder("nomic-embed-text", client=_client_with_handler(handler))
        vector = await embedder.embed_query("some text")
        assert list(vector) == pytest.approx([0.1, 0.2, 0.3])

    async def test_dimensions_are_discovered_from_the_response(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"embedding": [0.0] * 768})

        embedder = OllamaEmbedder("nomic-embed-text", client=_client_with_handler(handler))
        await embedder.embed_query("text")
        assert embedder.dimensions == 768

    async def test_dimensions_raises_before_any_call_and_no_explicit_value_given(self):
        embedder = OllamaEmbedder("m")
        with pytest.raises(RuntimeError):
            _ = embedder.dimensions

    async def test_explicit_dimensions_available_immediately(self):
        embedder = OllamaEmbedder("m", dimensions=384)
        assert embedder.dimensions == 384

    async def test_connection_error_raises_runtime_unavailable(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused", request=request)

        embedder = OllamaEmbedder("m", client=_client_with_handler(handler))
        with pytest.raises(RuntimeUnavailable):
            await embedder.embed_query("text")

    async def test_unexpected_response_shape_raises_invalid_model_response(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"not_embedding": "oops"})

        embedder = OllamaEmbedder("m", client=_client_with_handler(handler))
        with pytest.raises(InvalidModelResponse):
            await embedder.embed_query("text")

    async def test_empty_embedding_list_raises_invalid_model_response(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"embedding": []})

        embedder = OllamaEmbedder("m", client=_client_with_handler(handler))
        with pytest.raises(InvalidModelResponse):
            await embedder.embed_query("text")


class TestOllamaEmbedderEmbedDocuments:
    async def test_returns_one_vector_per_text(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"embedding": [1.0, 0.0]})

        embedder = OllamaEmbedder("m", client=_client_with_handler(handler))
        vectors = await embedder.embed_documents(["a", "b", "c"])
        assert len(vectors) == 3

    async def test_sends_the_model_name_in_the_request(self):
        received_models = []

        def handler(request: httpx.Request) -> httpx.Response:
            import json

            received_models.append(json.loads(request.content)["model"])
            return httpx.Response(200, json={"embedding": [1.0]})

        embedder = OllamaEmbedder("nomic-embed-text", client=_client_with_handler(handler))
        await embedder.embed_documents(["text"])
        assert received_models == ["nomic-embed-text"]

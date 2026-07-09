import numpy as np
import pytest

from local_ai_rag.embeddings.sentence_transformers_embedder import SentenceTransformersEmbedder


def fake_load_fn(model_name: str):
    return f"loaded-model:{model_name}"


def fake_encode_fn(model, texts: list[str]):
    # Deterministic: length-based fake vectors, 2 dims per text.
    return [np.array([float(len(t)), float(len(t.split()))]) for t in texts]


class TestEmbedDocuments:
    async def test_returns_one_vector_per_text(self):
        embedder = SentenceTransformersEmbedder("model-a", load_fn=fake_load_fn, encode_fn=fake_encode_fn)
        vectors = await embedder.embed_documents(["hello world", "a"])
        assert len(vectors) == 2

    async def test_loads_the_model_only_once_across_calls(self):
        load_calls = []

        def counting_load_fn(model_name):
            load_calls.append(model_name)
            return fake_load_fn(model_name)

        embedder = SentenceTransformersEmbedder("model-a", load_fn=counting_load_fn, encode_fn=fake_encode_fn)
        await embedder.embed_documents(["a"])
        await embedder.embed_documents(["b"])
        assert load_calls == ["model-a"]

    async def test_applies_the_document_prefix(self):
        received_texts = []

        def recording_encode_fn(model, texts):
            received_texts.extend(texts)
            return fake_encode_fn(model, texts)

        embedder = SentenceTransformersEmbedder(
            "model-a", load_fn=fake_load_fn, encode_fn=recording_encode_fn, document_prefix="passage: "
        )
        await embedder.embed_documents(["hello"])
        assert received_texts == ["passage: hello"]

    async def test_sets_dimensions_after_first_call(self):
        embedder = SentenceTransformersEmbedder("model-a", load_fn=fake_load_fn, encode_fn=fake_encode_fn)
        await embedder.embed_documents(["hello world"])
        assert embedder.dimensions == 2


class TestEmbedQuery:
    async def test_returns_a_single_vector(self):
        embedder = SentenceTransformersEmbedder("model-a", load_fn=fake_load_fn, encode_fn=fake_encode_fn)
        vector = await embedder.embed_query("hello world")
        assert isinstance(vector, np.ndarray)

    async def test_applies_the_query_prefix_not_the_document_prefix(self):
        received_texts = []

        def recording_encode_fn(model, texts):
            received_texts.extend(texts)
            return fake_encode_fn(model, texts)

        embedder = SentenceTransformersEmbedder(
            "model-a",
            load_fn=fake_load_fn,
            encode_fn=recording_encode_fn,
            query_prefix="query: ",
            document_prefix="passage: ",
        )
        await embedder.embed_query("hello")
        assert received_texts == ["query: hello"]


class TestDimensions:
    def test_raises_before_any_embed_call(self):
        embedder = SentenceTransformersEmbedder("model-a", load_fn=fake_load_fn, encode_fn=fake_encode_fn)
        with pytest.raises(RuntimeError):
            _ = embedder.dimensions

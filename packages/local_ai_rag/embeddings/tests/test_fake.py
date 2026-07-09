import numpy as np
import pytest

from local_ai_rag.embeddings.embedder import cosine_similarity
from local_ai_rag.embeddings.fake import FakeEmbedder


class TestConstruction:
    def test_rejects_nonpositive_dimensions(self):
        with pytest.raises(ValueError):
            FakeEmbedder(dimensions=0)

    def test_dimensions_property_matches_constructor_arg(self):
        embedder = FakeEmbedder(dimensions=16)
        assert embedder.dimensions == 16


class TestDeterminism:
    async def test_same_text_produces_the_same_vector(self):
        embedder = FakeEmbedder()
        v1 = await embedder.embed_query("hello world")
        v2 = await embedder.embed_query("hello world")
        assert np.array_equal(v1, v2)

    async def test_embed_documents_and_embed_query_agree_for_the_same_text(self):
        embedder = FakeEmbedder()
        doc_vecs = await embedder.embed_documents(["hello world"])
        query_vec = await embedder.embed_query("hello world")
        assert np.array_equal(doc_vecs[0], query_vec)


class TestVectorShape:
    async def test_returns_unit_length_vectors(self):
        embedder = FakeEmbedder(dimensions=8)
        v = await embedder.embed_query("some text here")
        assert np.linalg.norm(v) == pytest.approx(1.0)

    async def test_vector_length_matches_dimensions(self):
        embedder = FakeEmbedder(dimensions=12)
        v = await embedder.embed_query("text")
        assert len(v) == 12

    async def test_empty_string_does_not_crash(self):
        embedder = FakeEmbedder()
        v = await embedder.embed_query("")
        assert len(v) == embedder.dimensions


class TestSemanticPlausibility:
    async def test_texts_sharing_words_are_more_similar_than_unrelated_texts(self):
        embedder = FakeEmbedder(dimensions=64)
        a = await embedder.embed_query("the cat sat on the mat")
        b = await embedder.embed_query("the cat sat on the rug")  # shares 5/6 words
        c = await embedder.embed_query("quantum physics is fascinating")  # shares nothing
        assert cosine_similarity(a, b) > cosine_similarity(a, c)

    async def test_identical_texts_have_similarity_one(self):
        embedder = FakeEmbedder()
        a = await embedder.embed_query("identical text")
        b = await embedder.embed_query("identical text")
        assert cosine_similarity(a, b) == pytest.approx(1.0)


class TestCallCounting:
    async def test_tracks_embed_documents_calls(self):
        embedder = FakeEmbedder()
        await embedder.embed_documents(["a", "b"])
        await embedder.embed_documents(["c"])
        assert embedder.embed_documents_call_count == 2

    async def test_tracks_embed_query_calls(self):
        embedder = FakeEmbedder()
        await embedder.embed_query("a")
        await embedder.embed_query("b")
        assert embedder.embed_query_call_count == 2

    async def test_embed_documents_returns_one_vector_per_text(self):
        embedder = FakeEmbedder()
        vectors = await embedder.embed_documents(["one", "two", "three"])
        assert len(vectors) == 3

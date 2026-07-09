import numpy as np
import pytest

from local_ai_rag.stores.numpy_store import NumpyVectorStore


class TestAddAndSearch:
    async def test_search_returns_the_most_similar_document_first(self):
        store = NumpyVectorStore()
        await store.add("close", "close doc", np.array([1.0, 0.1]))
        await store.add("far", "far doc", np.array([0.0, 1.0]))
        results = await store.search(np.array([1.0, 0.0]), k=2)
        assert results[0].doc_id == "close"

    async def test_overwrites_on_same_doc_id(self):
        store = NumpyVectorStore()
        await store.add("d1", "first", np.array([1.0, 0.0]))
        await store.add("d1", "second", np.array([0.0, 1.0]))
        assert await store.count() == 1


class TestDelete:
    async def test_removes_the_document(self):
        store = NumpyVectorStore()
        await store.add("d1", "text", np.array([1.0, 0.0]))
        await store.delete("d1")
        assert await store.count() == 0

    async def test_missing_id_is_not_an_error(self):
        store = NumpyVectorStore()
        await store.delete("nope")
        assert await store.count() == 0


class TestMetadataFilter:
    async def test_excludes_nonmatching_documents(self):
        store = NumpyVectorStore()
        await store.add("d1", "doc 1", np.array([1.0, 0.0]), metadata={"tenant": "a"})
        await store.add("d2", "doc 2", np.array([1.0, 0.0]), metadata={"tenant": "b"})
        results = await store.search(np.array([1.0, 0.0]), k=5, metadata_filter={"tenant": "a"})
        assert [r.doc_id for r in results] == ["d1"]


class TestCount:
    async def test_reflects_number_of_documents(self):
        store = NumpyVectorStore()
        assert await store.count() == 0
        await store.add("d1", "text", np.array([1.0, 0.0]))
        assert await store.count() == 1


class TestSearchValidation:
    async def test_rejects_nonpositive_k(self):
        store = NumpyVectorStore()
        with pytest.raises(ValueError):
            await store.search(np.array([1.0]), k=0)

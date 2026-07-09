import numpy as np
import pytest

from local_ai_rag.stores.lancedb_store import LanceDBVectorStore


def make_store(tmp_path, name: str, dimensions: int = 2) -> LanceDBVectorStore:
    return LanceDBVectorStore(name, path=str(tmp_path / "lancedb"), dimensions=dimensions)


class TestAddAndSearch:
    async def test_search_returns_the_most_similar_document_first(self, tmp_path):
        store = make_store(tmp_path, "docs")
        await store.add("close", "close doc", np.array([1.0, 0.1]))
        await store.add("far", "far doc", np.array([0.0, 1.0]))
        results = await store.search(np.array([1.0, 0.0]), k=2)
        assert results[0].doc_id == "close"

    async def test_identical_vector_scores_close_to_one(self, tmp_path):
        store = make_store(tmp_path, "docs")
        await store.add("d1", "text", np.array([1.0, 0.0]))
        results = await store.search(np.array([1.0, 0.0]), k=1)
        assert results[0].score == pytest.approx(1.0, abs=1e-4)

    async def test_search_respects_k(self, tmp_path):
        store = make_store(tmp_path, "docs")
        for i in range(5):
            await store.add(f"d{i}", f"text {i}", np.array([1.0, float(i)]))
        results = await store.search(np.array([1.0, 0.0]), k=3)
        assert len(results) == 3

    async def test_search_result_carries_text_and_metadata(self, tmp_path):
        store = make_store(tmp_path, "docs")
        await store.add("d1", "the document text", np.array([1.0, 0.0]), metadata={"source": "manual"})
        results = await store.search(np.array([1.0, 0.0]), k=1)
        assert results[0].text == "the document text"
        assert results[0].metadata == {"source": "manual"}

    async def test_search_on_empty_store_returns_empty_list(self, tmp_path):
        store = make_store(tmp_path, "docs")
        assert await store.search(np.array([1.0, 0.0])) == []

    async def test_overwrites_on_same_doc_id(self, tmp_path):
        store = make_store(tmp_path, "docs")
        await store.add("d1", "first version", np.array([1.0, 0.0]))
        await store.add("d1", "second version", np.array([0.0, 1.0]))
        assert await store.count() == 1
        results = await store.search(np.array([0.0, 1.0]), k=1)
        assert results[0].text == "second version"

    async def test_rejects_nonpositive_k(self, tmp_path):
        store = make_store(tmp_path, "docs")
        with pytest.raises(ValueError):
            await store.search(np.array([1.0, 0.0]), k=0)


class TestDelete:
    async def test_removes_the_document(self, tmp_path):
        store = make_store(tmp_path, "docs")
        await store.add("d1", "text", np.array([1.0, 0.0]))
        await store.delete("d1")
        assert await store.count() == 0

    async def test_deleted_document_is_absent_from_search_results(self, tmp_path):
        store = make_store(tmp_path, "docs")
        await store.add("d1", "doc 1", np.array([1.0, 0.0]))
        await store.add("d2", "doc 2", np.array([1.0, 0.0]))
        await store.delete("d1")
        results = await store.search(np.array([1.0, 0.0]), k=5)
        assert [r.doc_id for r in results] == ["d2"]


class TestMetadataFilter:
    async def test_excludes_nonmatching_documents(self, tmp_path):
        store = make_store(tmp_path, "docs")
        await store.add("d1", "doc 1", np.array([1.0, 0.0]), metadata={"tenant": "a"})
        await store.add("d2", "doc 2", np.array([1.0, 0.0]), metadata={"tenant": "b"})
        results = await store.search(np.array([1.0, 0.0]), k=5, metadata_filter={"tenant": "a"})
        assert [r.doc_id for r in results] == ["d1"]

    async def test_requires_all_keys_to_match(self, tmp_path):
        store = make_store(tmp_path, "docs")
        await store.add("d1", "doc 1", np.array([1.0, 0.0]), metadata={"tenant": "a", "lang": "en"})
        await store.add("d2", "doc 2", np.array([1.0, 0.0]), metadata={"tenant": "a", "lang": "fr"})
        results = await store.search(
            np.array([1.0, 0.0]), k=5, metadata_filter={"tenant": "a", "lang": "en"}
        )
        assert [r.doc_id for r in results] == ["d1"]


class TestCount:
    async def test_reflects_number_of_documents(self, tmp_path):
        store = make_store(tmp_path, "docs")
        assert await store.count() == 0
        await store.add("d1", "text", np.array([1.0, 0.0]))
        assert await store.count() == 1


class TestPersistence:
    async def test_persists_across_a_new_connection_against_the_same_path(self, tmp_path):
        path = str(tmp_path / "lancedb")
        store1 = LanceDBVectorStore("persisted", path=path, dimensions=2)
        await store1.add("d1", "text", np.array([1.0, 0.0]))

        store2 = LanceDBVectorStore("persisted", path=path, dimensions=2)
        assert await store2.count() == 1

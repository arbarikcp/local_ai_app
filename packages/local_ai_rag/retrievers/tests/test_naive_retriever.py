from local_ai_rag.embeddings.fake import FakeEmbedder
from local_ai_rag.retrievers.naive_retriever import NaiveRetriever
from local_ai_rag.stores.numpy_store import NumpyVectorStore


class TestRetrieve:
    async def test_returns_the_most_similar_document_first(self):
        embedder = FakeEmbedder(dimensions=32)
        store = NumpyVectorStore()
        await store.add("d1", "how to reset your password", await embedder.embed_query("how to reset your password"))
        await store.add("d2", "distant galaxies and stars", await embedder.embed_query("distant galaxies and stars"))
        retriever = NaiveRetriever(embedder, store)

        results = await retriever.retrieve("I forgot my password", k=2)
        assert results[0].doc_id == "d1"

    async def test_respects_k(self):
        embedder = FakeEmbedder(dimensions=32)
        store = NumpyVectorStore()
        for i in range(5):
            await store.add(f"d{i}", f"document number {i}", await embedder.embed_query(f"document number {i}"))
        retriever = NaiveRetriever(embedder, store)

        results = await retriever.retrieve("document", k=3)
        assert len(results) == 3

    async def test_passes_through_metadata_filter(self):
        embedder = FakeEmbedder(dimensions=32)
        store = NumpyVectorStore()
        await store.add("d1", "billing info", await embedder.embed_query("billing info"), metadata={"category": "billing"})
        await store.add("d2", "shipping info", await embedder.embed_query("shipping info"), metadata={"category": "shipping"})
        retriever = NaiveRetriever(embedder, store)

        results = await retriever.retrieve("info", k=5, metadata_filter={"category": "billing"})
        assert [r.doc_id for r in results] == ["d1"]

    async def test_empty_store_returns_empty_list(self):
        embedder = FakeEmbedder(dimensions=32)
        store = NumpyVectorStore()
        retriever = NaiveRetriever(embedder, store)

        results = await retriever.retrieve("anything")
        assert results == []

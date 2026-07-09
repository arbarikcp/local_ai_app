from local_ai_rag.embeddings.fake import FakeEmbedder
from local_ai_rag.incremental_indexer import IncrementalIndexer
from local_ai_rag.loaders.markdown_loader import Document
from local_ai_rag.stores.numpy_store import NumpyVectorStore


def make_doc(doc_id: str, text: str) -> Document:
    return Document(doc_id=doc_id, source_path=f"/tmp/{doc_id}.md", title="T", text=text)


class TestDiff:
    def test_first_sync_treats_every_document_as_added(self):
        embedder = FakeEmbedder(dimensions=32)
        store = NumpyVectorStore()
        indexer = IncrementalIndexer(embedder, store)
        diff = indexer.diff([make_doc("a", "text a"), make_doc("b", "text b")])
        assert set(diff.added) == {"a", "b"}
        assert diff.updated == []
        assert diff.deleted == []


class TestSync:
    async def test_unchanged_document_is_not_re_embedded(self):
        embedder = FakeEmbedder(dimensions=32)
        store = NumpyVectorStore()
        indexer = IncrementalIndexer(embedder, store)
        docs = [make_doc("a", "text a"), make_doc("b", "text b")]
        await indexer.sync(docs)
        calls_after_first_sync = embedder.embed_documents_call_count

        diff = await indexer.sync(docs)
        assert diff.unchanged == ["a", "b"]
        assert embedder.embed_documents_call_count == calls_after_first_sync

    async def test_changed_document_is_re_embedded_and_old_chunks_removed(self):
        embedder = FakeEmbedder(dimensions=32)
        store = NumpyVectorStore()
        indexer = IncrementalIndexer(embedder, store, chunk_max_chars=1000)
        await indexer.sync([make_doc("a", "original text")])
        original_count = await store.count()

        diff = await indexer.sync([make_doc("a", "completely different revised text")])
        assert diff.updated == ["a"]
        assert await store.count() == original_count  # replaced, not accumulated

    async def test_removed_document_has_its_chunks_deleted(self):
        embedder = FakeEmbedder(dimensions=32)
        store = NumpyVectorStore()
        indexer = IncrementalIndexer(embedder, store)
        await indexer.sync([make_doc("a", "text a"), make_doc("b", "text b")])

        diff = await indexer.sync([make_doc("a", "text a")])
        assert diff.deleted == ["b"]
        results = await store.search((await embedder.embed_query("text b")), k=10)
        assert all(r.metadata["doc_id"] != "b" for r in results)

    async def test_a_newly_added_document_is_indexed(self):
        embedder = FakeEmbedder(dimensions=32)
        store = NumpyVectorStore()
        indexer = IncrementalIndexer(embedder, store)
        await indexer.sync([make_doc("a", "text a")])
        diff = await indexer.sync([make_doc("a", "text a"), make_doc("b", "text b")])
        assert diff.added == ["b"]
        assert await store.count() == 2

    async def test_content_hash_manifest_survives_across_calls(self):
        embedder = FakeEmbedder(dimensions=32)
        store = NumpyVectorStore()
        indexer = IncrementalIndexer(embedder, store)
        await indexer.sync([make_doc("a", "text a")])
        diff = await indexer.sync([make_doc("a", "text a")])
        assert diff.unchanged == ["a"]

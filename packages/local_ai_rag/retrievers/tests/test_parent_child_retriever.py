from local_ai_rag.chunkers.parent_child_chunker import chunk_document_parent_child
from local_ai_rag.embeddings.fake import FakeEmbedder
from local_ai_rag.loaders.markdown_loader import Document
from local_ai_rag.retrievers.parent_child_retriever import ParentChildRetriever
from local_ai_rag.stores.numpy_store import NumpyVectorStore

LONG_TEXT = (
    "Password reset links expire in fifteen minutes for security reasons. "
    "You can request a new link from the sign-in page at any time.\n\n"
    "Billing is charged monthly or annually depending on your chosen plan. "
    "Annual plans receive a twenty percent discount compared to monthly billing."
)


def make_doc() -> Document:
    return Document(doc_id="handbook", source_path="/tmp/handbook.md", title="T", text=LONG_TEXT)


async def build_retriever():
    embedder = FakeEmbedder(dimensions=64)
    store = NumpyVectorStore()
    index = chunk_document_parent_child(make_doc(), parent_max_chars=200, child_max_chars=60)
    for child in index.children:
        vector = await embedder.embed_query(child.text)
        await store.add(child.chunk_id, child.text, vector, metadata={"parent_id": child.parent_id})
    return ParentChildRetriever(embedder, store, index), index


class TestRetrieve:
    async def test_returns_parent_text_not_child_text(self):
        retriever, index = await build_retriever()
        results = await retriever.retrieve("password reset link expiry", k=2)
        assert any(r.text in index.parents[r.parent_id].text for r in results)

    async def test_deduplicates_multiple_child_hits_from_the_same_parent(self):
        retriever, index = await build_retriever()
        results = await retriever.retrieve("password reset", k=5, fetch_k=20)
        parent_ids = [r.parent_id for r in results]
        assert len(parent_ids) == len(set(parent_ids))

    async def test_most_relevant_parent_is_returned_first(self):
        retriever, index = await build_retriever()
        results = await retriever.retrieve("password reset link expiry", k=2)
        top_parent_text = index.parents[results[0].parent_id].text
        assert "password" in top_parent_text.lower()

    async def test_respects_k(self):
        retriever, index = await build_retriever()
        results = await retriever.retrieve("billing", k=1)
        assert len(results) <= 1

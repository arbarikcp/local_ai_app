from local_ai_rag.chunkers.document_chunker import chunk_document, chunk_documents
from local_ai_rag.loaders.markdown_loader import Document


def make_doc(doc_id: str, text: str) -> Document:
    return Document(doc_id=doc_id, source_path=f"/tmp/{doc_id}.md", title="T", text=text)


class TestChunkDocument:
    def test_chunk_ids_are_doc_id_and_index_joined(self):
        doc = make_doc("d1", "short text")
        chunks = chunk_document(doc, max_chars=1000)
        assert chunks[0].chunk_id == "d1::0"

    def test_every_chunk_carries_the_source_doc_id(self):
        doc = make_doc("d1", "para one.\n\npara two.\n\npara three.")
        chunks = chunk_document(doc, max_chars=15)
        assert all(c.doc_id == "d1" for c in chunks)

    def test_long_text_produces_multiple_chunks_with_increasing_index(self):
        doc = make_doc("d1", "para one.\n\npara two.\n\npara three.\n\npara four.")
        chunks = chunk_document(doc, max_chars=12)
        assert len(chunks) > 1
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))

    def test_short_text_produces_a_single_chunk(self):
        doc = make_doc("d1", "one short paragraph")
        chunks = chunk_document(doc, max_chars=1000)
        assert len(chunks) == 1
        assert chunks[0].text == "one short paragraph"


class TestChunkDocuments:
    def test_chunk_ids_stay_unique_across_documents(self):
        docs = [make_doc("d1", "text one"), make_doc("d2", "text two")]
        chunks = chunk_documents(docs, max_chars=1000)
        chunk_ids = [c.chunk_id for c in chunks]
        assert len(chunk_ids) == len(set(chunk_ids))

    def test_empty_document_list_produces_no_chunks(self):
        assert chunk_documents([], max_chars=1000) == []

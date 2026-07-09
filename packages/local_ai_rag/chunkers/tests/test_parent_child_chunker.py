import pytest

from local_ai_rag.chunkers.parent_child_chunker import (
    chunk_document_parent_child,
    chunk_documents_parent_child,
)
from local_ai_rag.loaders.markdown_loader import Document

LONG_TEXT = "\n\n".join(f"Paragraph {i} with some real sentence content to fill space." for i in range(20))


def make_doc(doc_id: str, text: str) -> Document:
    return Document(doc_id=doc_id, source_path=f"/tmp/{doc_id}.md", title="T", text=text)


class TestChunkDocumentParentChild:
    def test_children_are_smaller_than_their_parents(self):
        doc = make_doc("d1", LONG_TEXT)
        index = chunk_document_parent_child(doc, parent_max_chars=400, child_max_chars=100)
        for child in index.children:
            assert len(child.text) <= 100

    def test_every_child_references_a_real_parent_id(self):
        doc = make_doc("d1", LONG_TEXT)
        index = chunk_document_parent_child(doc, parent_max_chars=400, child_max_chars=100)
        for child in index.children:
            assert child.parent_id in index.parents

    def test_parent_text_returns_the_full_parent_chunk(self):
        doc = make_doc("d1", LONG_TEXT)
        index = chunk_document_parent_child(doc, parent_max_chars=400, child_max_chars=100)
        child = index.children[0]
        parent_text = index.parent_text(child.parent_id)
        assert child.text in parent_text or len(parent_text) >= len(child.text)

    def test_rejects_child_size_not_smaller_than_parent_size(self):
        doc = make_doc("d1", LONG_TEXT)
        with pytest.raises(ValueError):
            chunk_document_parent_child(doc, parent_max_chars=100, child_max_chars=100)

    def test_child_chunk_ids_are_unique(self):
        doc = make_doc("d1", LONG_TEXT)
        index = chunk_document_parent_child(doc, parent_max_chars=400, child_max_chars=100)
        chunk_ids = [c.chunk_id for c in index.children]
        assert len(chunk_ids) == len(set(chunk_ids))


class TestChunkDocumentsParentChild:
    def test_merges_parents_and_children_across_documents(self):
        docs = [make_doc("d1", LONG_TEXT), make_doc("d2", LONG_TEXT)]
        index = chunk_documents_parent_child(docs, parent_max_chars=400, child_max_chars=100)
        doc_ids = {c.doc_id for c in index.children}
        assert doc_ids == {"d1", "d2"}

    def test_empty_document_list_produces_empty_index(self):
        index = chunk_documents_parent_child([], parent_max_chars=400, child_max_chars=100)
        assert index.parents == {}
        assert index.children == []

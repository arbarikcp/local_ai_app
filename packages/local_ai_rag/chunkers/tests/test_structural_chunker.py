from local_ai_rag.chunkers.structural_chunker import chunk_preserving_structure
from local_ai_rag.loaders.markdown_loader import Document

TABLE_TEXT = (
    "Intro paragraph before the table.\n\n"
    "| Name | Value |\n"
    "|------|-------|\n"
    "| a    | 1     |\n"
    "| b    | 2     |\n\n"
    "Trailing paragraph after the table."
)

CODE_TEXT = (
    "Intro paragraph before the code.\n\n"
    "```python\n"
    "def f():\n"
    "    return 1\n"
    "```\n\n"
    "Trailing paragraph after the code."
)


def make_doc(text: str) -> Document:
    return Document(doc_id="d1", source_path="/tmp/d1.md", title="T", text=text)


class TestTablePreservation:
    def test_the_table_is_never_split_across_chunks(self):
        doc = make_doc(TABLE_TEXT)
        chunks = chunk_preserving_structure(doc, max_chars=30)
        table_chunks = [c for c in chunks if "| a" in c.text]
        assert len(table_chunks) == 1
        assert "| b    | 2     |" in table_chunks[0].text

    def test_a_chunk_containing_the_table_is_flagged(self):
        doc = make_doc(TABLE_TEXT)
        chunks = chunk_preserving_structure(doc, max_chars=30)
        table_chunk = next(c for c in chunks if "| a" in c.text)
        assert table_chunk.contains_structural_block is True

    def test_prose_only_chunks_are_not_flagged(self):
        doc = make_doc(TABLE_TEXT)
        chunks = chunk_preserving_structure(doc, max_chars=30)
        prose_chunks = [c for c in chunks if "| a" not in c.text]
        assert any(not c.contains_structural_block for c in prose_chunks)


class TestCodeBlockPreservation:
    def test_the_code_block_is_never_split_across_chunks(self):
        doc = make_doc(CODE_TEXT)
        chunks = chunk_preserving_structure(doc, max_chars=20)
        code_chunks = [c for c in chunks if "def f():" in c.text]
        assert len(code_chunks) == 1
        assert "return 1" in code_chunks[0].text
        assert "```" in code_chunks[0].text

    def test_a_chunk_containing_code_is_flagged(self):
        doc = make_doc(CODE_TEXT)
        chunks = chunk_preserving_structure(doc, max_chars=20)
        code_chunk = next(c for c in chunks if "def f():" in c.text)
        assert code_chunk.contains_structural_block is True


class TestBothStructuresInOneDocument:
    def test_table_and_code_block_both_survive_intact(self):
        text = TABLE_TEXT + "\n\n" + CODE_TEXT
        doc = make_doc(text)
        chunks = chunk_preserving_structure(doc, max_chars=40)
        joined = "\n".join(c.text for c in chunks)
        assert "| b    | 2     |" in joined
        assert "def f():\n    return 1" in joined

    def test_chunk_ids_are_unique(self):
        text = TABLE_TEXT + "\n\n" + CODE_TEXT
        doc = make_doc(text)
        chunks = chunk_preserving_structure(doc, max_chars=40)
        chunk_ids = [c.chunk_id for c in chunks]
        assert len(chunk_ids) == len(set(chunk_ids))


class TestPlainProseUnaffected:
    def test_a_document_with_no_structure_chunks_normally(self):
        doc = make_doc("Just plain prose.\n\nAnother plain paragraph.")
        chunks = chunk_preserving_structure(doc, max_chars=1000)
        assert len(chunks) == 1
        assert chunks[0].contains_structural_block is False

from local_ai_rag.chunkers.semantic_chunker import chunk_semantically, split_sentences
from local_ai_rag.embeddings.fake import FakeEmbedder
from local_ai_rag.loaders.markdown_loader import Document

TOPIC_SHIFT_TEXT = (
    "The password reset link expires in fifteen minutes. "
    "The password reset process is easy to follow. "
    "Distant galaxies contain billions of stars. "
    "Astronomers study distant galaxies using telescopes."
)


def make_doc(text: str) -> Document:
    return Document(doc_id="d1", source_path="/tmp/d1.md", title="T", text=text)


class TestSplitSentences:
    def test_splits_on_sentence_boundaries(self):
        sentences = split_sentences("First sentence. Second sentence! Third sentence?")
        assert sentences == ["First sentence.", "Second sentence!", "Third sentence?"]

    def test_empty_text_returns_empty_list(self):
        assert split_sentences("") == []

    def test_collapses_internal_whitespace(self):
        sentences = split_sentences("One.\n\n  Two.")
        assert sentences == ["One.", "Two."]


class TestChunkSemantically:
    async def test_topic_shift_produces_a_new_chunk(self):
        embedder = FakeEmbedder(dimensions=64)
        doc = make_doc(TOPIC_SHIFT_TEXT)
        chunks = await chunk_semantically(doc, embedder, similarity_threshold=0.15)
        assert len(chunks) == 2

    async def test_related_sentences_stay_in_the_same_chunk(self):
        embedder = FakeEmbedder(dimensions=64)
        doc = make_doc(TOPIC_SHIFT_TEXT)
        chunks = await chunk_semantically(doc, embedder, similarity_threshold=0.15)
        assert "password reset link expires" in chunks[0].text
        assert "password reset process" in chunks[0].text

    async def test_unrelated_sentences_end_up_in_different_chunks(self):
        embedder = FakeEmbedder(dimensions=64)
        doc = make_doc(TOPIC_SHIFT_TEXT)
        chunks = await chunk_semantically(doc, embedder, similarity_threshold=0.15)
        assert "galaxies" not in chunks[0].text
        assert "password" not in chunks[1].text

    async def test_chunk_ids_are_stable_and_unique(self):
        embedder = FakeEmbedder(dimensions=64)
        doc = make_doc(TOPIC_SHIFT_TEXT)
        chunks = await chunk_semantically(embedder=embedder, document=doc, similarity_threshold=0.15)
        chunk_ids = [c.chunk_id for c in chunks]
        assert len(chunk_ids) == len(set(chunk_ids))
        assert all(c.doc_id == "d1" for c in chunks)

    async def test_empty_document_produces_no_chunks(self):
        embedder = FakeEmbedder(dimensions=64)
        doc = make_doc("")
        chunks = await chunk_semantically(doc, embedder)
        assert chunks == []

    async def test_a_low_threshold_merges_everything_into_one_chunk(self):
        embedder = FakeEmbedder(dimensions=64)
        doc = make_doc(TOPIC_SHIFT_TEXT)
        chunks = await chunk_semantically(doc, embedder, similarity_threshold=-1.0)
        assert len(chunks) == 1

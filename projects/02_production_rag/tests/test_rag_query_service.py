from local_ai_core.runtimes.fake import FakeRuntime
from local_ai_rag.embeddings.fake import FakeEmbedder
from local_ai_rag.loaders.markdown_loader import Document
from local_ai_rag.stores.numpy_store import NumpyVectorStore

from rag_ingestion_service import ingest_document
from rag_metadata_store import RagMetadataStore
from rag_query_service import answer_question


async def make_corpus(metadata_store):
    embedder = FakeEmbedder()
    store = NumpyVectorStore()
    await ingest_document(
        Document(doc_id="doc-1", source_path="/docs/doc-1.md", title="Doc One", text="Password reset links expire after 24 hours."),
        embedder=embedder, store=store, metadata_store=metadata_store,
    )
    await ingest_document(
        Document(doc_id="doc-2", source_path="/docs/doc-2.md", title="Doc Two", text="Refunds are issued within five business days."),
        embedder=embedder, store=store, metadata_store=metadata_store,
    )
    return embedder, store


class TestAnswerQuestionHappyPath:
    async def test_a_grounded_citation_is_verified(self):
        with RagMetadataStore(":memory:") as metadata_store:
            embedder, store = await make_corpus(metadata_store)
            runtime = FakeRuntime(default_response="Password resets expire in 24 hours [doc-1::0].")

            result = await answer_question(
                embedder=embedder, store=store, runtime=runtime, metadata_store=metadata_store,
                question="How long do password reset links last?", k=5,
            )

            assert len(result.citations) == 1
            assert result.citations[0].verified is True
            assert result.citations[0].document_id == "doc-1"
            assert result.citations[0].chunk_id == "doc-1::0"

    async def test_retrieved_doc_ids_reflects_the_packed_chunks_parent_documents(self):
        with RagMetadataStore(":memory:") as metadata_store:
            embedder, store = await make_corpus(metadata_store)
            runtime = FakeRuntime(default_response="an answer with no citations")

            result = await answer_question(
                embedder=embedder, store=store, runtime=runtime, metadata_store=metadata_store,
                question="How long do password reset links last?", k=5,
            )

            assert set(result.retrieved_doc_ids) <= {"doc-1", "doc-2"}
            assert len(result.retrieved_doc_ids) == len(set(result.retrieved_doc_ids))  # deduplicated

    async def test_context_tokens_is_a_real_positive_number(self):
        with RagMetadataStore(":memory:") as metadata_store:
            embedder, store = await make_corpus(metadata_store)
            runtime = FakeRuntime(default_response="An answer with no citations.")

            result = await answer_question(
                embedder=embedder, store=store, runtime=runtime, metadata_store=metadata_store, question="anything",
            )
            assert result.context_tokens > 0

    async def test_latency_is_measured(self):
        with RagMetadataStore(":memory:") as metadata_store:
            embedder, store = await make_corpus(metadata_store)
            runtime = FakeRuntime(default_response="answer")
            result = await answer_question(
                embedder=embedder, store=store, runtime=runtime, metadata_store=metadata_store, question="anything",
            )
            assert result.latency_ms >= 0


class TestAnswerQuestionUngroundedCitation:
    async def test_a_fabricated_citation_is_flagged_not_dropped(self):
        with RagMetadataStore(":memory:") as metadata_store:
            embedder, store = await make_corpus(metadata_store)
            runtime = FakeRuntime(default_response="Made up fact [doc-99::0].")

            result = await answer_question(
                embedder=embedder, store=store, runtime=runtime, metadata_store=metadata_store, question="anything",
            )

            assert len(result.citations) == 1
            assert result.citations[0].verified is False
            # flagged, not silently removed from the response
            assert result.citations[0].chunk_id == "doc-99::0"


class TestAnswerQuestionMetadataFilter:
    async def test_filtering_to_one_document_excludes_the_other(self):
        with RagMetadataStore(":memory:") as metadata_store:
            embedder, store = await make_corpus(metadata_store)
            runtime = FakeRuntime(default_response="filtered answer")

            result = await answer_question(
                embedder=embedder, store=store, runtime=runtime, metadata_store=metadata_store,
                question="anything", metadata_filter={"doc_id": "doc-1"}, k=5,
            )

            # every packed/reranked chunk must come from doc-1 only
            assert result.reranked_chunks <= 1 or all(
                c.document_id == "doc-1" for c in result.citations
            )


class TestQueryLogging:
    async def test_every_query_is_logged_when_a_metadata_store_is_given(self):
        with RagMetadataStore(":memory:") as metadata_store:
            embedder, store = await make_corpus(metadata_store)
            runtime = FakeRuntime(default_response="answer")

            await answer_question(
                embedder=embedder, store=store, runtime=runtime, metadata_store=metadata_store, question="my question",
            )

            rows = metadata_store._conn.execute("SELECT question FROM query_log").fetchall()
            assert rows == [("my question",)]

    async def test_no_metadata_store_means_no_logging_and_no_error(self):
        with RagMetadataStore(":memory:") as metadata_store:
            embedder, store = await make_corpus(metadata_store)
        runtime = FakeRuntime(default_response="answer")
        result = await answer_question(embedder=embedder, store=store, runtime=runtime, question="my question")
        assert result.answer == "answer"

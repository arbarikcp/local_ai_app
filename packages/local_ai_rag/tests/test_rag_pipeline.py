from local_ai_core.runtimes.fake import FakeRuntime
from local_ai_rag.embeddings.fake import FakeEmbedder
from local_ai_rag.loaders.markdown_loader import Document
from local_ai_rag.pipeline import NaiveRagPipeline
from local_ai_rag.stores.numpy_store import NumpyVectorStore

DOCS = [
    Document(
        doc_id="password_reset",
        source_path="/tmp/password_reset.md",
        title="Resetting Your Password",
        text="The password reset link expires in 15 minutes for security reasons.",
    ),
    Document(
        doc_id="billing_overview",
        source_path="/tmp/billing_overview.md",
        title="Billing Overview",
        text="Nimbus bills subscriptions monthly or annually, charged automatically.",
    ),
]


def make_pipeline(**runtime_kwargs) -> NaiveRagPipeline:
    embedder = FakeEmbedder(dimensions=32)
    store = NumpyVectorStore()
    runtime = FakeRuntime(**runtime_kwargs)
    return NaiveRagPipeline(embedder, store, runtime, model="fake-model", chunk_max_chars=500)


class TestIngestDocuments:
    async def test_returns_one_chunk_per_short_document(self):
        pipeline = make_pipeline()
        chunks = await pipeline.ingest_documents(DOCS)
        assert len(chunks) == 2

    async def test_populates_the_store(self):
        pipeline = make_pipeline()
        await pipeline.ingest_documents(DOCS)
        results = await pipeline.retrieve("password reset link expiry", k=2)
        assert len(results) == 2

    async def test_empty_document_list_ingests_nothing(self):
        pipeline = make_pipeline()
        chunks = await pipeline.ingest_documents([])
        assert chunks == []

    async def test_chunk_count_reflects_the_store(self):
        pipeline = make_pipeline()
        await pipeline.ingest_documents(DOCS)
        assert await pipeline.chunk_count() == 2


class TestRetrieve:
    async def test_returns_the_most_relevant_chunk_first(self):
        pipeline = make_pipeline()
        await pipeline.ingest_documents(DOCS)
        results = await pipeline.retrieve("when does my password reset link expire", k=1)
        assert results[0].doc_id == "password_reset::0"


class TestAnswer:
    async def test_answer_includes_the_runtimes_generated_text(self):
        pipeline = make_pipeline(default_response="The link expires in 15 minutes [password_reset::0].")
        await pipeline.ingest_documents(DOCS)
        result = await pipeline.answer("When does the password reset link expire?", k=2)
        assert "15 minutes" in result.answer_text

    async def test_extracts_citations_from_the_answer(self):
        pipeline = make_pipeline(default_response="15 minutes [password_reset::0].")
        await pipeline.ingest_documents(DOCS)
        result = await pipeline.answer("q", k=2)
        assert result.citations == ["password_reset::0"]

    async def test_grounded_citation_is_flagged_as_grounded(self):
        pipeline = make_pipeline(default_response="15 minutes [password_reset::0].")
        await pipeline.ingest_documents(DOCS)
        result = await pipeline.answer("q", k=2)
        assert result.citations_are_grounded is True

    async def test_invented_citation_is_flagged_as_not_grounded(self):
        pipeline = make_pipeline(default_response="Answer with a fake source [totally_made_up::9].")
        await pipeline.ingest_documents(DOCS)
        result = await pipeline.answer("q", k=2)
        assert result.citations_are_grounded is False

    async def test_no_citations_is_vacuously_grounded(self):
        pipeline = make_pipeline(default_response="I don't know based on the provided documents.")
        await pipeline.ingest_documents(DOCS)
        result = await pipeline.answer("q", k=2)
        assert result.citations == []
        assert result.citations_are_grounded is True

    async def test_retrieved_chunks_are_carried_on_the_answer(self):
        pipeline = make_pipeline()
        await pipeline.ingest_documents(DOCS)
        result = await pipeline.answer("password reset", k=2)
        assert len(result.retrieved_chunks) == 2

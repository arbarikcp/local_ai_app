from local_ai_core.deployment.config import AppConfig
from local_ai_core.runtimes.fake import FakeRuntime
from local_ai_rag.embeddings.fake import FakeEmbedder
from local_ai_rag.loaders.markdown_loader import Document

from rag_ingestion_service import ingest_document
from rag_query_service import answer_question
from rag_service import build_rag_context

REPO_ROOT_CATALOG = "models/MODEL_CATALOG.md"


def make_config(tmp_path) -> AppConfig:
    return AppConfig.model_validate(
        {
            "app": {"data_dir": str(tmp_path / "data")},
            "models": {
                "default_chat": "a",
                "default_extraction": "b",
                "default_code": "c",
                "default_embedding": "test-embedding-model",
            },
        }
    )


class TestBuildRagContext:
    def test_wires_a_real_persistent_vector_store_and_metadata_store(self, tmp_path):
        config = make_config(tmp_path)
        ctx = build_rag_context(config, model_catalog_path=REPO_ROOT_CATALOG)

        assert ctx.store is not None
        assert ctx.metadata_store is not None
        assert (ctx.base.data_dir.base_dir / "rag" / "rag_metadata.db").exists()
        assert (ctx.base.data_dir.base_dir / "rag" / "lancedb").exists()

    def test_defaults_to_a_fake_embedder(self, tmp_path):
        config = make_config(tmp_path)
        ctx = build_rag_context(config, model_catalog_path=REPO_ROOT_CATALOG)
        assert isinstance(ctx.embedder, FakeEmbedder)

    def test_accepts_an_injected_embedder(self, tmp_path):
        config = make_config(tmp_path)
        custom_embedder = FakeEmbedder(dimensions=64)
        ctx = build_rag_context(config, model_catalog_path=REPO_ROOT_CATALOG, embedder=custom_embedder)
        assert ctx.embedder is custom_embedder


class TestFullIngestAndQueryRoundTrip:
    async def test_a_document_ingested_through_the_composition_root_is_queryable(self, tmp_path):
        config = make_config(tmp_path)
        runtime = FakeRuntime(default_response="Password resets expire in 24 hours [doc-1::0].")
        ctx = build_rag_context(config, model_catalog_path=REPO_ROOT_CATALOG, runtime=runtime)

        await ingest_document(
            Document(
                doc_id="doc-1",
                source_path="/docs/doc-1.md",
                title="Doc One",
                text="Password reset links expire after 24 hours.",
            ),
            embedder=ctx.embedder,
            store=ctx.store,
            metadata_store=ctx.metadata_store,
        )

        result = await answer_question(
            embedder=ctx.embedder,
            store=ctx.store,
            runtime=ctx.base.runtime,
            metadata_store=ctx.metadata_store,
            question="How long do password reset links last?",
        )

        assert result.citations[0].verified is True
        assert result.citations[0].document_id == "doc-1"

    async def test_the_ingested_document_survives_a_real_close_and_reopen_of_the_metadata_store(self, tmp_path):
        config = make_config(tmp_path)
        ctx = build_rag_context(config, model_catalog_path=REPO_ROOT_CATALOG)

        await ingest_document(
            Document(doc_id="doc-1", source_path="/docs/doc-1.md", title="Doc One", text="Some real content."),
            embedder=ctx.embedder,
            store=ctx.store,
            metadata_store=ctx.metadata_store,
        )
        ctx.metadata_store.close()

        reopened = build_rag_context(config, model_catalog_path=REPO_ROOT_CATALOG)
        record = reopened.metadata_store.get_document("doc-1")

        assert record is not None
        assert record.status == "ingested"

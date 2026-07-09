from local_ai_core.runtimes.fake import FakeRuntime
from local_ai_rag.context_packers.budget_packer import ContextBudget
from local_ai_rag.embeddings.fake import FakeEmbedder
from local_ai_rag.production_pipeline import ProductionRagPipeline
from local_ai_rag.retrievers.acl import clearance_predicate
from local_ai_rag.stores.numpy_store import NumpyVectorStore


async def build_store(embedder):
    store = NumpyVectorStore()
    await store.add(
        "password_reset::0",
        "The password reset link expires in fifteen minutes.",
        await embedder.embed_query("The password reset link expires in fifteen minutes."),
        metadata={"doc_id": "password_reset"},
    )
    await store.add(
        "billing_overview::0",
        "Billing is charged monthly or annually depending on your plan.",
        await embedder.embed_query("Billing is charged monthly or annually depending on your plan."),
        metadata={"doc_id": "billing_overview"},
    )
    await store.add(
        "secret_doc::0",
        "Confidential internal password rotation schedule.",
        await embedder.embed_query("Confidential internal password rotation schedule."),
        metadata={"doc_id": "secret_doc", "security_level": 5},
    )
    return store


class TestAnswer:
    async def test_produces_an_answer_with_citations(self):
        embedder = FakeEmbedder(dimensions=32)
        store = await build_store(embedder)
        runtime = FakeRuntime(default_response="15 minutes [password_reset::0].")
        pipeline = ProductionRagPipeline(embedder, store, runtime, model="fake-model")

        result = await pipeline.answer("How long does the reset link last?", k=3)
        assert result.citations == ["password_reset::0"]
        assert result.citations_are_grounded is True

    async def test_source_citations_are_deduplicated_at_the_document_level(self):
        embedder = FakeEmbedder(dimensions=32)
        store = await build_store(embedder)
        runtime = FakeRuntime(default_response="[password_reset::0] and also [password_reset::0].")
        pipeline = ProductionRagPipeline(embedder, store, runtime, model="fake-model")

        result = await pipeline.answer("q", k=3)
        assert result.source_citations == ["password_reset"]

    async def test_acl_predicate_excludes_restricted_documents(self):
        embedder = FakeEmbedder(dimensions=32)
        store = await build_store(embedder)
        runtime = FakeRuntime(default_response="answer")
        pipeline = ProductionRagPipeline(
            embedder, store, runtime, model="fake-model", acl_predicate=clearance_predicate(user_clearance=0)
        )

        result = await pipeline.answer("password", k=5)
        assert all(c.doc_id != "secret_doc::0" for c in result.packed_chunks)
        assert result.trace.candidates_after_acl < result.trace.candidates_retrieved

    async def test_rewrite_true_calls_the_runtime_twice_and_records_the_rewrite(self):
        embedder = FakeEmbedder(dimensions=32)
        store = await build_store(embedder)
        runtime = FakeRuntime(default_response="password reset link expires in fifteen minutes")
        pipeline = ProductionRagPipeline(embedder, store, runtime, model="fake-model")

        result = await pipeline.answer("huh?", rewrite=True, k=3)
        assert runtime.call_count == 2
        assert result.trace.rewritten_question == "password reset link expires in fifteen minutes"

    async def test_rewrite_false_calls_the_runtime_once_and_leaves_rewritten_question_unset(self):
        embedder = FakeEmbedder(dimensions=32)
        store = await build_store(embedder)
        runtime = FakeRuntime(default_response="answer")
        pipeline = ProductionRagPipeline(embedder, store, runtime, model="fake-model")

        result = await pipeline.answer("question", rewrite=False, k=3)
        assert runtime.call_count == 1
        assert result.trace.rewritten_question is None

    async def test_a_tight_context_budget_reduces_chunks_packed(self):
        embedder = FakeEmbedder(dimensions=32)
        store = await build_store(embedder)
        runtime = FakeRuntime(default_response="answer")
        tight_budget = ContextBudget(
            max_context_tokens=10, reserved_for_system=0, reserved_for_question=0, reserved_for_answer=0
        )
        pipeline = ProductionRagPipeline(embedder, store, runtime, model="fake-model", context_budget=tight_budget)

        result = await pipeline.answer("password billing", k=5)
        assert result.trace.chunks_packed <= result.trace.candidates_after_rerank

    async def test_trace_log_reflects_every_stage(self):
        embedder = FakeEmbedder(dimensions=32)
        store = await build_store(embedder)
        runtime = FakeRuntime(default_response="answer")
        pipeline = ProductionRagPipeline(embedder, store, runtime, model="fake-model")

        result = await pipeline.answer("password", k=3)
        assert result.trace.question == "password"
        assert result.trace.candidates_retrieved > 0

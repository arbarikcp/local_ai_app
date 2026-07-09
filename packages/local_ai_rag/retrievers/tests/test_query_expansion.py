from local_ai_core.runtimes.fake import FakeRuntime
from local_ai_rag.embeddings.fake import FakeEmbedder
from local_ai_rag.retrievers.naive_retriever import NaiveRetriever
from local_ai_rag.retrievers.query_expansion import (
    generate_query_variants,
    hyde_retrieve,
    multi_query_retrieve,
    rewrite_and_retrieve,
    rewrite_query,
)
from local_ai_rag.stores.numpy_store import NumpyVectorStore


async def build_store_and_embedder():
    embedder = FakeEmbedder(dimensions=64)
    store = NumpyVectorStore()
    await store.add("d1", "password reset link expires in fifteen minutes", await embedder.embed_query("password reset link expires in fifteen minutes"))
    await store.add("d2", "billing is charged monthly or annually", await embedder.embed_query("billing is charged monthly or annually"))
    return embedder, store


class TestRewriteQuery:
    async def test_returns_the_runtimes_response_text_stripped(self):
        runtime = FakeRuntime(default_response="  How do I reset my password?  \n")
        rewritten = await rewrite_query("forgot my password", runtime, model="fake-model")
        assert rewritten == "How do I reset my password?"

    async def test_sends_a_request_for_the_configured_model(self):
        runtime = FakeRuntime(default_response="x")
        await rewrite_query("q", runtime, model="fake-model")
        assert runtime.requests_received[0].model == "fake-model"


class TestRewriteAndRetrieve:
    async def test_retrieves_using_the_rewritten_query(self):
        embedder, store = await build_store_and_embedder()
        retriever = NaiveRetriever(embedder, store)
        runtime = FakeRuntime(default_response="password reset link expires in fifteen minutes")
        results = await rewrite_and_retrieve("huh", retriever, runtime, model="fake-model", k=1)
        assert results[0].doc_id == "d1"


class TestGenerateQueryVariants:
    async def test_parses_one_variant_per_line(self):
        runtime = FakeRuntime(default_response="How do I reset my password?\nWhat is the password reset process?\nPassword recovery steps?")
        variants = await generate_query_variants("forgot password", runtime, model="fake-model", n=3)
        assert len(variants) == 3

    async def test_falls_back_to_the_original_question_if_the_response_is_empty(self):
        runtime = FakeRuntime(default_response="   \n  ")
        variants = await generate_query_variants("forgot password", runtime, model="fake-model", n=3)
        assert variants == ["forgot password"]


class TestMultiQueryRetrieve:
    async def test_fuses_results_across_variants(self):
        embedder, store = await build_store_and_embedder()
        retriever = NaiveRetriever(embedder, store)
        runtime = FakeRuntime(default_response="password reset\npassword recovery\nreset link")
        results = await multi_query_retrieve("forgot password", retriever, runtime, model="fake-model", n_queries=3, k=2)
        assert results[0].doc_id == "d1"

    async def test_respects_k(self):
        embedder, store = await build_store_and_embedder()
        retriever = NaiveRetriever(embedder, store)
        runtime = FakeRuntime(default_response="password\nbilling\nreset")
        results = await multi_query_retrieve("q", retriever, runtime, model="fake-model", n_queries=3, k=1)
        assert len(results) <= 1


class TestHydeRetrieve:
    async def test_retrieves_using_the_hypothetical_passage_embedding(self):
        embedder, store = await build_store_and_embedder()
        runtime = FakeRuntime(default_response="Password reset links expire in fifteen minutes for security.")
        results = await hyde_retrieve("how long until my link expires", embedder, store, runtime, model="fake-model", k=1)
        assert results[0].doc_id == "d1"

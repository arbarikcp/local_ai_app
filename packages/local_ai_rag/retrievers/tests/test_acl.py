from local_ai_rag.embeddings.fake import FakeEmbedder
from local_ai_rag.retrievers.acl import AclAwareRetriever, clearance_predicate, tenant_predicate
from local_ai_rag.retrievers.naive_retriever import NaiveRetriever
from local_ai_rag.stores.numpy_store import NumpyVectorStore


async def build_store_and_embedder():
    embedder = FakeEmbedder(dimensions=32)
    store = NumpyVectorStore()
    await store.add("public", "public policy document", await embedder.embed_query("public policy document"), metadata={"security_level": 0})
    await store.add("secret", "confidential policy document", await embedder.embed_query("confidential policy document"), metadata={"security_level": 5})
    await store.add("untagged", "untagged policy document", await embedder.embed_query("untagged policy document"), metadata={})
    return embedder, store


class TestClearancePredicate:
    async def test_low_clearance_user_only_sees_public_documents(self):
        embedder, store = await build_store_and_embedder()
        base = NaiveRetriever(embedder, store)
        acl_retriever = AclAwareRetriever(base, clearance_predicate(user_clearance=0))
        results = await acl_retriever.retrieve("policy document", k=5)
        doc_ids = {r.doc_id for r in results}
        assert "secret" not in doc_ids

    async def test_high_clearance_user_sees_everything(self):
        embedder, store = await build_store_and_embedder()
        base = NaiveRetriever(embedder, store)
        acl_retriever = AclAwareRetriever(base, clearance_predicate(user_clearance=10))
        results = await acl_retriever.retrieve("policy document", k=5)
        doc_ids = {r.doc_id for r in results}
        assert "secret" in doc_ids

    async def test_untagged_documents_are_treated_as_public(self):
        embedder, store = await build_store_and_embedder()
        base = NaiveRetriever(embedder, store)
        acl_retriever = AclAwareRetriever(base, clearance_predicate(user_clearance=0))
        results = await acl_retriever.retrieve("policy document", k=5)
        doc_ids = {r.doc_id for r in results}
        assert "untagged" in doc_ids

    async def test_over_fetching_keeps_k_results_when_some_are_filtered(self):
        embedder, store = await build_store_and_embedder()
        base = NaiveRetriever(embedder, store)
        acl_retriever = AclAwareRetriever(base, clearance_predicate(user_clearance=0), fetch_multiplier=4)
        results = await acl_retriever.retrieve("policy document", k=2)
        assert len(results) == 2


class TestTenantPredicate:
    async def test_only_matching_tenant_documents_are_returned(self):
        embedder = FakeEmbedder(dimensions=32)
        store = NumpyVectorStore()
        await store.add("a", "doc a", await embedder.embed_query("doc a"), metadata={"tenant_id": "acme"})
        await store.add("b", "doc b", await embedder.embed_query("doc b"), metadata={"tenant_id": "globex"})
        base = NaiveRetriever(embedder, store)
        acl_retriever = AclAwareRetriever(base, tenant_predicate("acme"))
        results = await acl_retriever.retrieve("doc", k=5)
        assert [r.doc_id for r in results] == ["a"]

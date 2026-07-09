import numpy as np
import pytest

from local_ai_rag.stores.hybrid import (
    hybrid_search,
    keyword_score,
    keyword_search,
    reciprocal_rank_fusion,
)
from local_ai_rag.stores.numpy_store import NumpyVectorStore


class TestKeywordScore:
    def test_shared_words_score_above_zero(self):
        assert keyword_score("reset password", "how to reset your password") > 0.0

    def test_unrelated_text_scores_zero(self):
        assert keyword_score("reset password", "distant galaxies and stars") == 0.0

    def test_empty_query_scores_zero(self):
        assert keyword_score("", "some document text") == 0.0

    def test_empty_document_scores_zero(self):
        assert keyword_score("reset password", "") == 0.0

    def test_full_overlap_scores_one(self):
        assert keyword_score("reset password", "reset password") == pytest.approx(1.0)


class TestKeywordSearch:
    def test_ranks_the_exact_term_match_first(self):
        documents = {
            "d1": "order ACC88213 status update",
            "d2": "generic unrelated document about weather",
        }
        results = keyword_search("ACC88213", documents, k=5)
        assert results[0][0] == "d1"

    def test_excludes_documents_with_zero_score(self):
        documents = {"d1": "reset password", "d2": "distant galaxies"}
        results = keyword_search("reset password", documents, k=5)
        assert [doc_id for doc_id, _score in results] == ["d1"]

    def test_respects_k(self):
        documents = {f"d{i}": "reset password" for i in range(10)}
        results = keyword_search("reset password", documents, k=3)
        assert len(results) == 3


class TestReciprocalRankFusion:
    def test_document_in_both_lists_outranks_document_in_one(self):
        fused = reciprocal_rank_fusion([["a", "b"], ["a", "c"]])
        ranked_ids = [doc_id for doc_id, _score in fused]
        assert ranked_ids[0] == "a"

    def test_includes_documents_from_either_list(self):
        fused = reciprocal_rank_fusion([["a"], ["b"]])
        ranked_ids = {doc_id for doc_id, _score in fused}
        assert ranked_ids == {"a", "b"}

    def test_empty_lists_produce_no_results(self):
        assert reciprocal_rank_fusion([[], []]) == []


class TestHybridSearch:
    async def test_recovers_a_document_vector_search_alone_would_rank_last(self):
        # d_code shares an exact term with the query but has a vector far from it;
        # d_close is vector-nearest but shares no words with the query.
        store = NumpyVectorStore()
        await store.add("d_close", "unrelated content", np.array([1.0, 0.0]))
        await store.add("d_code", "your order ACC88213 has shipped", np.array([0.0, 1.0]))
        documents = {
            "d_close": "unrelated content",
            "d_code": "your order ACC88213 has shipped",
        }

        query_embedding = np.array([1.0, 0.0])  # closest to d_close, not d_code
        vector_only = await store.search(query_embedding, k=1)
        assert vector_only[0].doc_id == "d_close"  # vector search alone misses d_code

        hybrid_results = await hybrid_search(
            store, documents, query="ACC88213", query_embedding=query_embedding, k=2
        )
        assert "d_code" in [r.doc_id for r in hybrid_results]

    async def test_returns_at_most_k_results(self):
        store = NumpyVectorStore()
        documents = {}
        for i in range(5):
            text = f"document number {i}"
            await store.add(f"d{i}", text, np.array([1.0, float(i)]))
            documents[f"d{i}"] = text
        results = await hybrid_search(
            store, documents, query="document", query_embedding=np.array([1.0, 0.0]), k=2
        )
        assert len(results) <= 2

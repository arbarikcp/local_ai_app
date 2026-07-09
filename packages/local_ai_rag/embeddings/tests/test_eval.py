import math

import pytest

from local_ai_rag.embeddings.embedder import NumpyEmbeddingIndex
from local_ai_rag.embeddings.eval import (
    EmbeddingEvalCase,
    evaluate_embedder,
    measure_embedding_throughput,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from local_ai_rag.embeddings.fake import FakeEmbedder


class TestRecallAtK:
    def test_all_relevant_docs_retrieved_is_full_recall(self):
        assert recall_at_k(["d1", "d2", "d3"], {"d1", "d2"}, k=3) == 1.0

    def test_none_retrieved_is_zero_recall(self):
        assert recall_at_k(["d3", "d4"], {"d1", "d2"}, k=2) == 0.0

    def test_partial_recall(self):
        assert recall_at_k(["d1", "d3"], {"d1", "d2"}, k=2) == pytest.approx(0.5)

    def test_only_considers_top_k(self):
        # d1 is relevant but ranked 3rd - k=2 shouldn't see it.
        assert recall_at_k(["d9", "d8", "d1"], {"d1"}, k=2) == 0.0

    def test_no_relevant_docs_returns_zero_not_a_crash(self):
        assert recall_at_k(["d1"], set(), k=1) == 0.0


class TestPrecisionAtK:
    def test_all_top_k_relevant_is_full_precision(self):
        assert precision_at_k(["d1", "d2"], {"d1", "d2"}, k=2) == 1.0

    def test_none_relevant_is_zero_precision(self):
        assert precision_at_k(["d3", "d4"], {"d1"}, k=2) == 0.0

    def test_partial_precision(self):
        assert precision_at_k(["d1", "d3"], {"d1"}, k=2) == pytest.approx(0.5)

    def test_empty_retrieved_list_returns_zero(self):
        assert precision_at_k([], {"d1"}, k=5) == 0.0


class TestReciprocalRank:
    def test_first_result_relevant_gives_rr_one(self):
        assert reciprocal_rank(["d1", "d2"], {"d1"}) == 1.0

    def test_second_result_relevant_gives_rr_half(self):
        assert reciprocal_rank(["d2", "d1"], {"d1"}) == pytest.approx(0.5)

    def test_no_relevant_result_gives_rr_zero(self):
        assert reciprocal_rank(["d2", "d3"], {"d1"}) == 0.0

    def test_uses_the_first_relevant_hit_only(self):
        assert reciprocal_rank(["d2", "d1", "d1"], {"d1"}) == pytest.approx(0.5)


class TestNdcgAtK:
    def test_perfect_ranking_gives_ndcg_one(self):
        # Both relevant docs ranked first - DCG should equal IDCG.
        assert ndcg_at_k(["d1", "d2"], {"d1", "d2"}, k=2) == pytest.approx(1.0)

    def test_worse_ranking_gives_lower_ndcg_than_perfect(self):
        perfect = ndcg_at_k(["d1", "d2", "d3"], {"d1", "d2"}, k=3)
        worse = ndcg_at_k(["d3", "d1", "d2"], {"d1", "d2"}, k=3)
        assert worse < perfect

    def test_no_relevant_docs_returns_zero(self):
        assert ndcg_at_k(["d1"], set(), k=1) == 0.0

    def test_manual_calculation_for_a_known_case(self):
        # relevant = {d1}, retrieved = [d2, d1] -> DCG = 1/log2(3), IDCG = 1/log2(2) = 1
        result = ndcg_at_k(["d2", "d1"], {"d1"}, k=2)
        expected = (1.0 / math.log2(3)) / 1.0
        assert result == pytest.approx(expected)


class TestEvaluateEmbedder:
    async def test_produces_one_result_per_eval_case(self):
        embedder = FakeEmbedder()
        index = NumpyEmbeddingIndex()
        index.add("d1", "the cat sat on the mat", await embedder.embed_query("the cat sat on the mat"))
        index.add("d2", "unrelated text about space", await embedder.embed_query("unrelated text about space"))

        cases = [EmbeddingEvalCase(query="cat mat", positive_doc_ids=["d1"])]
        summary = await evaluate_embedder(embedder, index, cases, k=2)
        assert len(summary.case_results) == 1

    async def test_relevant_document_is_found_when_words_overlap(self):
        embedder = FakeEmbedder(dimensions=64)
        index = NumpyEmbeddingIndex()
        cat_text = "the cat sat on the mat"
        space_text = "distant galaxies and stars"
        index.add("cat-doc", cat_text, await embedder.embed_query(cat_text))
        index.add("space-doc", space_text, await embedder.embed_query(space_text))

        cases = [EmbeddingEvalCase(query="the cat sat on a mat", positive_doc_ids=["cat-doc"])]
        summary = await evaluate_embedder(embedder, index, cases, k=1)
        assert summary.mean_recall_at_k == 1.0

    async def test_summary_properties_average_across_cases(self):
        embedder = FakeEmbedder()
        index = NumpyEmbeddingIndex()
        index.add("d1", "text", await embedder.embed_query("text"))

        cases = [
            EmbeddingEvalCase(query="text", positive_doc_ids=["d1"]),
            EmbeddingEvalCase(query="text", positive_doc_ids=["not-indexed"]),
        ]
        summary = await evaluate_embedder(embedder, index, cases, k=1)
        assert summary.mean_recall_at_k == pytest.approx(0.5)  # one hit, one miss

    async def test_empty_eval_cases_returns_zeroed_summary(self):
        embedder = FakeEmbedder()
        index = NumpyEmbeddingIndex()
        summary = await evaluate_embedder(embedder, index, [], k=5)
        assert summary.mean_recall_at_k == 0.0
        assert summary.mrr == 0.0

    async def test_latency_is_recorded_per_case(self):
        embedder = FakeEmbedder()
        index = NumpyEmbeddingIndex()
        index.add("d1", "text", await embedder.embed_query("text"))
        cases = [EmbeddingEvalCase(query="text", positive_doc_ids=["d1"])]
        summary = await evaluate_embedder(embedder, index, cases, k=1)
        assert summary.case_results[0].latency_seconds >= 0
        assert summary.mean_query_latency_seconds >= 0


class TestMeasureEmbeddingThroughput:
    async def test_returns_a_positive_rate_for_nonempty_input(self):
        embedder = FakeEmbedder()
        rate = await measure_embedding_throughput(embedder, ["a", "b", "c"])
        assert rate > 0

    async def test_empty_input_returns_zero(self):
        embedder = FakeEmbedder()
        assert await measure_embedding_throughput(embedder, []) == 0.0

    async def test_calls_embed_documents_exactly_once(self):
        embedder = FakeEmbedder()
        await measure_embedding_throughput(embedder, ["a", "b"])
        assert embedder.embed_documents_call_count == 1

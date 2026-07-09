import math

import pytest

from local_ai_core.evals.retrieval_metrics import (
    context_precision,
    context_recall,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


class TestRecallAtK:
    def test_all_relevant_docs_retrieved_is_full_recall(self):
        assert recall_at_k(["d1", "d2", "d3"], {"d1", "d2"}, k=3) == 1.0

    def test_no_relevant_docs_returns_zero_not_a_crash(self):
        assert recall_at_k(["d1"], set(), k=1) == 0.0


class TestPrecisionAtK:
    def test_partial_precision(self):
        assert precision_at_k(["d1", "d3"], {"d1"}, k=2) == pytest.approx(0.5)

    def test_empty_retrieved_list_returns_zero(self):
        assert precision_at_k([], {"d1"}, k=5) == 0.0


class TestReciprocalRank:
    def test_second_result_relevant_gives_rr_half(self):
        assert reciprocal_rank(["d2", "d1"], {"d1"}) == pytest.approx(0.5)


class TestNdcgAtK:
    def test_manual_calculation_for_a_known_case(self):
        result = ndcg_at_k(["d2", "d1"], {"d1"}, k=2)
        expected = (1.0 / math.log2(3)) / 1.0
        assert result == pytest.approx(expected)


class TestRagasStyleAliases:
    def test_context_precision_is_precision_at_k(self):
        assert context_precision is precision_at_k

    def test_context_recall_is_recall_at_k(self):
        assert context_recall is recall_at_k

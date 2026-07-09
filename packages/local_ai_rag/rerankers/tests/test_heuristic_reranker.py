import pytest

from local_ai_rag.embeddings.embedder import SearchResult
from local_ai_rag.rerankers.heuristic_reranker import HeuristicReranker


def make_result(doc_id: str, score: float, text: str) -> SearchResult:
    return SearchResult(doc_id=doc_id, score=score, text=text, metadata={})


class TestRerank:
    def test_a_low_vector_score_exact_keyword_match_can_move_to_the_top(self):
        candidates = [
            make_result("vector_favorite", 0.9, "unrelated content sharing no words"),
            make_result("keyword_match", 0.1, "your order ACC88213 has shipped"),
        ]
        reranker = HeuristicReranker(keyword_weight=0.9)
        results = reranker.rerank("ACC88213", candidates)
        assert results[0].doc_id == "keyword_match"

    def test_zero_keyword_weight_preserves_the_original_vector_ordering(self):
        candidates = [make_result("a", 0.9, "text a"), make_result("b", 0.5, "text b")]
        reranker = HeuristicReranker(keyword_weight=0.0)
        results = reranker.rerank("query", candidates)
        assert [r.doc_id for r in results] == ["a", "b"]

    def test_respects_k(self):
        candidates = [make_result(f"d{i}", 1.0 - i * 0.1, f"text {i}") for i in range(5)]
        reranker = HeuristicReranker(keyword_weight=0.5)
        results = reranker.rerank("text", candidates, k=2)
        assert len(results) == 2

    def test_empty_candidates_returns_empty_list(self):
        reranker = HeuristicReranker()
        assert reranker.rerank("query", []) == []

    def test_rejects_out_of_range_keyword_weight(self):
        with pytest.raises(ValueError):
            HeuristicReranker(keyword_weight=1.5)

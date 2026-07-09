import pytest

from local_ai_rag.context_packers.budget_packer import (
    ContextBudget,
    estimate_tokens,
    order_for_generation,
    pack_context,
)
from local_ai_rag.embeddings.embedder import SearchResult


def make_result(doc_id: str, text: str, score: float = 1.0, source: str | None = None) -> SearchResult:
    metadata = {"doc_id": source or doc_id}
    return SearchResult(doc_id=doc_id, score=score, text=text, metadata=metadata)


class TestEstimateTokens:
    def test_returns_at_least_one_for_nonempty_text(self):
        assert estimate_tokens("hi") >= 1

    def test_scales_roughly_with_word_count(self):
        assert estimate_tokens("one two three four five six seven eight") > estimate_tokens("one two")


class TestContextBudget:
    def test_available_for_chunks_matches_the_curriculum_example(self):
        budget = ContextBudget(
            max_context_tokens=6000, reserved_for_system=500, reserved_for_question=300, reserved_for_answer=1000
        )
        assert budget.available_for_chunks == 4200

    def test_available_for_chunks_never_goes_negative(self):
        budget = ContextBudget(
            max_context_tokens=100, reserved_for_system=200, reserved_for_question=0, reserved_for_answer=0
        )
        assert budget.available_for_chunks == 0

    def test_rejects_negative_fields(self):
        with pytest.raises(ValueError):
            ContextBudget(max_context_tokens=-1, reserved_for_system=0, reserved_for_question=0, reserved_for_answer=0)


class TestPackContext:
    def test_packs_within_the_token_budget(self):
        budget = ContextBudget(max_context_tokens=20, reserved_for_system=0, reserved_for_question=0, reserved_for_answer=0)
        candidates = [make_result(f"d{i}", "word " * 10, source=f"doc{i}") for i in range(5)]
        packed = pack_context(candidates, budget)
        total_tokens = sum(estimate_tokens(c.text) for c in packed)
        assert total_tokens <= budget.available_for_chunks

    def test_a_later_shorter_candidate_still_fits_after_a_longer_one_is_skipped(self):
        budget = ContextBudget(max_context_tokens=10, reserved_for_system=0, reserved_for_question=0, reserved_for_answer=0)
        candidates = [
            make_result("long", "word " * 20, source="doc1"),
            make_result("short", "one two", source="doc2"),
        ]
        packed = pack_context(candidates, budget)
        assert [c.doc_id for c in packed] == ["short"]

    def test_source_diversity_caps_chunks_from_one_source(self):
        budget = ContextBudget(max_context_tokens=1000, reserved_for_system=0, reserved_for_question=0, reserved_for_answer=0)
        candidates = [make_result(f"d{i}", "short text", source="same_doc") for i in range(5)]
        packed = pack_context(candidates, budget, max_chunks_per_source=2)
        assert len(packed) == 2

    def test_empty_candidates_returns_empty_list(self):
        budget = ContextBudget(max_context_tokens=100, reserved_for_system=0, reserved_for_question=0, reserved_for_answer=0)
        assert pack_context([], budget) == []


class TestOrderForGeneration:
    def test_highest_relevance_chunk_is_first(self):
        packed = [make_result("a", "text"), make_result("b", "text"), make_result("c", "text")]
        ordered = order_for_generation(packed)
        assert ordered[0].doc_id == "a"

    def test_second_highest_relevance_chunk_is_last(self):
        packed = [make_result("a", "text"), make_result("b", "text"), make_result("c", "text")]
        ordered = order_for_generation(packed)
        assert ordered[-1].doc_id == "b"

    def test_weakest_chunk_ends_up_in_the_middle(self):
        packed = [make_result(x, "text") for x in ["a", "b", "c", "d", "e"]]
        ordered = order_for_generation(packed)
        assert ordered[2].doc_id == "e"

    def test_preserves_all_chunks(self):
        packed = [make_result(x, "text") for x in ["a", "b", "c", "d"]]
        ordered = order_for_generation(packed)
        assert {c.doc_id for c in ordered} == {"a", "b", "c", "d"}

    def test_empty_input_returns_empty_list(self):
        assert order_for_generation([]) == []

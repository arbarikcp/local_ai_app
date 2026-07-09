from datetime import datetime, timedelta, timezone

import pytest

from local_ai_rag.embeddings.embedder import SearchResult
from local_ai_rag.retrievers.time_aware import apply_recency_boost

NOW = datetime(2026, 7, 9, tzinfo=timezone.utc)


def make_result(doc_id: str, score: float, created_at: str | None) -> SearchResult:
    metadata = {"created_at": created_at} if created_at is not None else {}
    return SearchResult(doc_id=doc_id, score=score, text="text", metadata=metadata)


class TestApplyRecencyBoost:
    def test_a_document_at_exactly_the_half_life_has_its_score_halved(self):
        created_at = (NOW - timedelta(days=30)).isoformat()
        results = [make_result("d1", 1.0, created_at)]
        boosted = apply_recency_boost(results, half_life_days=30.0, now=NOW)
        assert boosted[0].score == pytest.approx(0.5)

    def test_a_brand_new_document_keeps_its_original_score(self):
        results = [make_result("d1", 0.8, NOW.isoformat())]
        boosted = apply_recency_boost(results, half_life_days=30.0, now=NOW)
        assert boosted[0].score == pytest.approx(0.8)

    def test_a_document_missing_the_timestamp_is_left_unboosted(self):
        results = [make_result("d1", 0.9, None)]
        boosted = apply_recency_boost(results, half_life_days=30.0, now=NOW)
        assert boosted[0].score == 0.9

    def test_an_old_but_more_relevant_document_can_still_outrank_a_recent_weak_match(self):
        old_relevant = make_result("old", 1.0, (NOW - timedelta(days=1)).isoformat())
        recent_weak = make_result("recent", 0.05, NOW.isoformat())
        boosted = apply_recency_boost([recent_weak, old_relevant], half_life_days=30.0, now=NOW)
        assert boosted[0].doc_id == "old"

    def test_a_very_old_document_is_boosted_below_a_comparably_relevant_recent_one(self):
        old = make_result("old", 1.0, (NOW - timedelta(days=365)).isoformat())
        recent = make_result("recent", 1.0, NOW.isoformat())
        boosted = apply_recency_boost([old, recent], half_life_days=30.0, now=NOW)
        assert boosted[0].doc_id == "recent"

    def test_results_are_sorted_by_boosted_score_descending(self):
        a = make_result("a", 0.5, NOW.isoformat())
        b = make_result("b", 0.9, NOW.isoformat())
        boosted = apply_recency_boost([a, b], half_life_days=30.0, now=NOW)
        assert [r.doc_id for r in boosted] == ["b", "a"]

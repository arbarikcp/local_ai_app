import pytest

from local_ai_agents.planners.loop_prevention import LoopDetectedError, LoopGuard


class TestRecord:
    def test_distinct_calls_never_trip_the_guard(self):
        guard = LoopGuard(max_repeats=3)
        guard.record("calculator", {"expression": "1+1"})
        guard.record("calculator", {"expression": "2+2"})
        guard.record("calculator", {"expression": "3+3"})

    def test_trips_after_max_repeats_identical_calls_in_a_row(self):
        guard = LoopGuard(max_repeats=3)
        guard.record("calculator", {"expression": "2+2"})
        guard.record("calculator", {"expression": "2+2"})
        with pytest.raises(LoopDetectedError):
            guard.record("calculator", {"expression": "2+2"})

    def test_a_different_call_in_between_resets_the_count(self):
        guard = LoopGuard(max_repeats=2)
        guard.record("calculator", {"expression": "2+2"})
        guard.record("search_files", {"query": "x"})
        guard.record("calculator", {"expression": "2+2"})  # not consecutive with the first
        assert guard.consecutive_count == 1

    def test_argument_order_does_not_affect_the_signature(self):
        guard = LoopGuard(max_repeats=2)
        guard.record("search_files", {"query": "x", "max_results": 5})
        with pytest.raises(LoopDetectedError):
            guard.record("search_files", {"max_results": 5, "query": "x"})

    def test_consecutive_count_tracks_the_current_streak(self):
        guard = LoopGuard(max_repeats=5)
        guard.record("calculator", {"expression": "1"})
        guard.record("calculator", {"expression": "1"})
        assert guard.consecutive_count == 2

    def test_rejects_a_nonpositive_max_repeats(self):
        with pytest.raises(ValueError):
            LoopGuard(max_repeats=0)

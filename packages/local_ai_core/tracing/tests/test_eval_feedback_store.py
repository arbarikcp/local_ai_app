import pytest

from local_ai_core.tracing.eval_feedback_store import (
    EvalFeedbackStore,
    EvalRunRecord,
    UserFeedbackRecord,
)


class TestEvalRuns:
    def test_logged_eval_run_can_be_retrieved_by_trace_id(self):
        with EvalFeedbackStore() as store:
            store.log_eval_run(EvalRunRecord(trace_id="trace-1", metric_name="must_contain_score", score=0.8))
            runs = store.get_eval_runs_for_trace("trace-1")
            assert len(runs) == 1
            assert runs[0].metric_name == "must_contain_score"
            assert runs[0].score == 0.8
            assert runs[0].created_at is not None

    def test_multiple_eval_runs_for_the_same_trace_are_kept_in_order(self):
        with EvalFeedbackStore() as store:
            store.log_eval_run(EvalRunRecord(trace_id="trace-1", metric_name="must_contain_score", score=0.8))
            store.log_eval_run(EvalRunRecord(trace_id="trace-1", metric_name="refusal_check", score=1.0))
            runs = store.get_eval_runs_for_trace("trace-1")
            assert [r.metric_name for r in runs] == ["must_contain_score", "refusal_check"]

    def test_a_trace_with_no_eval_runs_returns_an_empty_list(self):
        with EvalFeedbackStore() as store:
            assert store.get_eval_runs_for_trace("no-such-trace") == []


class TestUserFeedback:
    def test_logged_feedback_can_be_retrieved_by_trace_id(self):
        with EvalFeedbackStore() as store:
            store.log_user_feedback(UserFeedbackRecord(trace_id="trace-2", rating="up", comment="great answer"))
            feedback = store.get_feedback_for_trace("trace-2")
            assert len(feedback) == 1
            assert feedback[0].rating == "up"
            assert feedback[0].comment == "great answer"

    def test_invalid_rating_raises(self):
        with EvalFeedbackStore() as store:
            with pytest.raises(ValueError):
                store.log_user_feedback(UserFeedbackRecord(trace_id="trace-2", rating="sideways"))

    def test_feedback_summary_counts_by_rating(self):
        with EvalFeedbackStore() as store:
            store.log_user_feedback(UserFeedbackRecord(trace_id="trace-1", rating="up"))
            store.log_user_feedback(UserFeedbackRecord(trace_id="trace-2", rating="up"))
            store.log_user_feedback(UserFeedbackRecord(trace_id="trace-3", rating="down"))
            summary = store.feedback_summary()
            assert summary == {"up": 2, "down": 1}


class TestPersistsAcrossCloseAndReopen:
    def test_eval_run_survives_a_real_close_and_reopen_cycle(self, tmp_path):
        db_path = tmp_path / "eval_feedback.db"

        store = EvalFeedbackStore(db_path)
        store.log_eval_run(EvalRunRecord(trace_id="trace-1", metric_name="must_contain_score", score=0.8))
        store.close()

        reopened = EvalFeedbackStore(db_path)
        runs = reopened.get_eval_runs_for_trace("trace-1")
        reopened.close()

        assert len(runs) == 1
        assert runs[0].score == 0.8

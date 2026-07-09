import pytest

from local_ai_core.conversation.summarizer import summarize_then_truncate
from local_ai_core.conversation.token_budget import ConversationBudget
from local_ai_core.conversation.turn import Turn


def _budget(context_window: int) -> ConversationBudget:
    return ConversationBudget(
        context_window=context_window, reserved_system=0, reserved_tools=0,
        reserved_current_user_turn=0, reserved_answer=0,
    )


def _one_token_counter(turns):
    return len(turns)


def _fake_summarize_fn(turns):
    return f"summary of {len(turns)} turns"


class TestSummarizeThenTruncate:
    def test_returns_unchanged_when_under_budget(self):
        turns = [Turn(role="user", content="a"), Turn(role="assistant", content="b")]
        result = summarize_then_truncate(turns, _budget(10), _fake_summarize_fn, token_counter=_one_token_counter)
        assert result == turns

    def test_rejects_negative_keep_last_n_raw(self):
        with pytest.raises(ValueError):
            summarize_then_truncate([], _budget(0), _fake_summarize_fn, keep_last_n_raw=-1)

    def test_summarizes_older_turns_and_keeps_last_n_raw(self):
        turns = [Turn(role="user", content=f"turn {i}") for i in range(6)]
        result = summarize_then_truncate(
            turns, _budget(3), _fake_summarize_fn, token_counter=_one_token_counter, keep_last_n_raw=2
        )
        # 1 summary turn + last 2 raw turns = 3 turns, fits the budget of 3.
        assert len(result) == 3
        assert result[-2:] == turns[-2:]
        assert "summary" in result[0].content

    def test_summary_turn_mentions_how_many_turns_it_covers(self):
        turns = [Turn(role="user", content=f"turn {i}") for i in range(6)]
        result = summarize_then_truncate(
            turns, _budget(3), _fake_summarize_fn, token_counter=_one_token_counter, keep_last_n_raw=2
        )
        # 6 turns total, 2 kept raw -> 4 summarized
        assert "summary of 4 turns" in result[0].content

    def test_always_keeps_sticky_turns_regardless_of_position(self):
        sticky = Turn(role="system", content="be terse", sticky=True)
        turns = [sticky] + [Turn(role="user", content=f"turn {i}") for i in range(6)]
        result = summarize_then_truncate(
            turns, _budget(3), _fake_summarize_fn, token_counter=_one_token_counter, keep_last_n_raw=1
        )
        assert sticky in result

    def test_never_splits_a_tool_call_result_group_when_summarizing(self):
        call = Turn(role="assistant", content="call", turn_group_id="g1")
        tool_result = Turn(role="tool", content="result", turn_group_id="g1")
        turns = [call, tool_result] + [Turn(role="user", content=f"turn {i}") for i in range(4)]
        result = summarize_then_truncate(
            turns, _budget(3), _fake_summarize_fn, token_counter=_one_token_counter, keep_last_n_raw=1
        )
        # the group is summarized as a unit - either both turns show up
        # summarized away together, or (if kept raw) both present.
        assert (call in result) == (tool_result in result)

    def test_short_conversation_with_nothing_old_enough_falls_back_to_drop_oldest(self):
        # keep_last_n_raw >= number of non-sticky groups means nothing is
        # old enough to summarize - falls back to drop_oldest instead of
        # doing nothing while still over budget.
        turns = [Turn(role="user", content=f"turn {i}") for i in range(3)]
        result = summarize_then_truncate(
            turns, _budget(2), _fake_summarize_fn, token_counter=_one_token_counter, keep_last_n_raw=5
        )
        assert len(result) == 2
        assert result == turns[-2:]

    def test_summarize_fn_receives_exactly_the_turns_being_compressed(self):
        received = []

        def recording_summarize_fn(turns_to_summarize):
            received.extend(turns_to_summarize)
            return "a summary"

        turns = [Turn(role="user", content=f"turn {i}") for i in range(5)]
        summarize_then_truncate(
            turns, _budget(2), recording_summarize_fn, token_counter=_one_token_counter, keep_last_n_raw=1
        )
        assert received == turns[:-1]  # everything except the last kept-raw turn

    def test_still_over_budget_after_summarizing_falls_back_to_drop_oldest(self):
        # Budget of 1 "token": even after collapsing everything old into one
        # summary turn + keeping 1 raw turn, that's 2 turns - still over
        # budget of 1, so drop_oldest must run as the final safety net.
        turns = [Turn(role="user", content=f"turn {i}") for i in range(5)]
        result = summarize_then_truncate(
            turns, _budget(1), _fake_summarize_fn, token_counter=_one_token_counter, keep_last_n_raw=1
        )
        assert len(result) == 1
        assert result == [turns[-1]]  # drop_oldest drops the summary, keeps the newest raw turn

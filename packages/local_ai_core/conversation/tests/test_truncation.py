import pytest

from local_ai_core.conversation.token_budget import ConversationBudget
from local_ai_core.conversation.truncation import drop_oldest, group_turns, keep_system_plus_last_n
from local_ai_core.conversation.turn import Turn


class TestGroupTurns:
    def test_ungrouped_turns_are_singleton_groups(self):
        turns = [Turn(role="user", content="a"), Turn(role="assistant", content="b")]
        groups = group_turns(turns)
        assert groups == [[turns[0]], [turns[1]]]

    def test_turns_sharing_a_group_id_are_combined(self):
        call = Turn(role="assistant", content="calling tool", turn_group_id="g1")
        result = Turn(role="tool", content="tool result", turn_group_id="g1")
        groups = group_turns([call, result])
        assert groups == [[call, result]]

    def test_mixed_grouped_and_ungrouped_turns(self):
        user = Turn(role="user", content="hi")
        call = Turn(role="assistant", content="calling", turn_group_id="g1")
        result = Turn(role="tool", content="result", turn_group_id="g1")
        reply = Turn(role="assistant", content="here you go")
        groups = group_turns([user, call, result, reply])
        assert groups == [[user], [call, result], [reply]]

    def test_empty_list_returns_empty_list(self):
        assert group_turns([]) == []


class TestDropOldest:
    def _budget(self, context_window: int) -> ConversationBudget:
        return ConversationBudget(
            context_window=context_window, reserved_system=0, reserved_tools=0,
            reserved_current_user_turn=0, reserved_answer=0,
        )

    def _one_token_counter(self, turns):
        return len(turns)  # one "token" per turn - makes budget math exact and simple

    def test_returns_all_turns_when_under_budget(self):
        turns = [Turn(role="user", content="a"), Turn(role="assistant", content="b")]
        result = drop_oldest(turns, self._budget(10), token_counter=self._one_token_counter)
        assert result == turns

    def test_drops_oldest_turns_first(self):
        turns = [Turn(role="user", content=f"turn {i}") for i in range(5)]
        result = drop_oldest(turns, self._budget(3), token_counter=self._one_token_counter)
        assert len(result) == 3
        assert result == turns[-3:]

    def test_never_drops_sticky_turns(self):
        sticky = Turn(role="system", content="system prompt", sticky=True)
        turns = [sticky] + [Turn(role="user", content=f"turn {i}") for i in range(5)]
        result = drop_oldest(turns, self._budget(2), token_counter=self._one_token_counter)
        assert sticky in result

    def test_stops_dropping_when_only_sticky_remains_even_if_still_over_budget(self):
        sticky = [Turn(role="system", content="a", sticky=True), Turn(role="system", content="b", sticky=True)]
        result = drop_oldest(sticky, self._budget(1), token_counter=self._one_token_counter)
        assert result == sticky  # can't go below 2 "tokens" without dropping sticky content

    def test_never_splits_a_tool_call_result_group(self):
        call = Turn(role="assistant", content="call", turn_group_id="g1")
        tool_result = Turn(role="tool", content="result", turn_group_id="g1")
        turns = [Turn(role="user", content="a"), call, tool_result]
        # Budget of 2 "tokens" - the group counts as 2 turns, so dropping it
        # entirely (not partially) is the only way to shrink below the group's size.
        result = drop_oldest(turns, self._budget(2), token_counter=self._one_token_counter)
        assert (call in result) == (tool_result in result)  # both present or both absent, never one


class TestKeepSystemPlusLastN:
    def test_keeps_last_n_groups_in_order(self):
        turns = [Turn(role="user", content=f"turn {i}") for i in range(5)]
        result = keep_system_plus_last_n(turns, n=2)
        assert result == turns[-2:]

    def test_always_keeps_sticky_turns_regardless_of_position(self):
        sticky = Turn(role="system", content="system prompt", sticky=True)
        turns = [sticky] + [Turn(role="user", content=f"turn {i}") for i in range(5)]
        result = keep_system_plus_last_n(turns, n=1)
        assert sticky in result
        assert turns[-1] in result
        assert len(result) == 2

    def test_n_zero_keeps_only_sticky_turns(self):
        sticky = Turn(role="system", content="system prompt", sticky=True)
        turns = [sticky, Turn(role="user", content="a"), Turn(role="user", content="b")]
        result = keep_system_plus_last_n(turns, n=0)
        assert result == [sticky]

    def test_rejects_negative_n(self):
        with pytest.raises(ValueError):
            keep_system_plus_last_n([], n=-1)

    def test_never_splits_a_tool_call_result_group(self):
        call = Turn(role="assistant", content="call", turn_group_id="g1")
        tool_result = Turn(role="tool", content="result", turn_group_id="g1")
        turns = [Turn(role="user", content="old")] + [call, tool_result]
        result = keep_system_plus_last_n(turns, n=1)
        assert call in result and tool_result in result

    def test_preserves_original_order_not_sticky_first(self):
        sticky = Turn(role="system", content="sys", sticky=True)
        turns = [Turn(role="user", content="early"), sticky, Turn(role="user", content="late")]
        result = keep_system_plus_last_n(turns, n=1)
        assert result == [sticky, turns[-1]]  # original relative order preserved

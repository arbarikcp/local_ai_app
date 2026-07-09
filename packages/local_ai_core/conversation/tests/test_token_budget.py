import pytest

from local_ai_core.conversation.token_budget import (
    ConversationBudget,
    heuristic_token_counter,
    history_exceeds_budget,
)
from local_ai_core.conversation.turn import Turn


class TestConversationBudget:
    def test_history_budget_subtracts_every_reservation(self):
        budget = ConversationBudget(
            context_window=8000, reserved_system=500, reserved_tools=500,
            reserved_current_user_turn=200, reserved_answer=800,
        )
        assert budget.history_budget == 8000 - 500 - 500 - 200 - 800

    def test_history_budget_never_goes_negative(self):
        budget = ConversationBudget(
            context_window=1000, reserved_system=500, reserved_tools=500,
            reserved_current_user_turn=500, reserved_answer=500,
        )
        assert budget.history_budget == 0

    def test_rejects_negative_fields(self):
        with pytest.raises(ValueError):
            ConversationBudget(
                context_window=-1, reserved_system=0, reserved_tools=0,
                reserved_current_user_turn=0, reserved_answer=0,
            )

    def test_zero_reservations_leave_full_context_window(self):
        budget = ConversationBudget(
            context_window=4000, reserved_system=0, reserved_tools=0,
            reserved_current_user_turn=0, reserved_answer=0,
        )
        assert budget.history_budget == 4000


class TestHeuristicTokenCounter:
    def test_counts_across_all_turns(self):
        turns = [Turn(role="user", content="one two three"), Turn(role="assistant", content="four five")]
        count = heuristic_token_counter(turns)
        assert count == pytest.approx(round(5 * 1.3))

    def test_empty_turns_is_zero(self):
        assert heuristic_token_counter([]) == 0

    def test_more_turns_means_more_tokens(self):
        short = heuristic_token_counter([Turn(role="user", content="hi")])
        long = heuristic_token_counter([Turn(role="user", content="a much longer message with many words")])
        assert long > short


class TestHistoryExceedsBudget:
    def test_false_when_under_budget(self):
        budget = ConversationBudget(
            context_window=1000, reserved_system=0, reserved_tools=0,
            reserved_current_user_turn=0, reserved_answer=0,
        )
        turns = [Turn(role="user", content="short")]
        assert history_exceeds_budget(turns, budget) is False

    def test_true_when_over_budget(self):
        budget = ConversationBudget(
            context_window=5, reserved_system=0, reserved_tools=0,
            reserved_current_user_turn=0, reserved_answer=0,
        )
        turns = [Turn(role="user", content="this message has way more than five tokens in it")]
        assert history_exceeds_budget(turns, budget) is True

    def test_uses_injected_token_counter(self):
        budget = ConversationBudget(
            context_window=10, reserved_system=0, reserved_tools=0,
            reserved_current_user_turn=0, reserved_answer=0,
        )

        def always_zero(turns):
            return 0

        turns = [Turn(role="user", content="anything at all, doesn't matter")]
        assert history_exceeds_budget(turns, budget, token_counter=always_zero) is False

    def test_exactly_at_budget_is_not_exceeding(self):
        budget = ConversationBudget(
            context_window=10, reserved_system=0, reserved_tools=0,
            reserved_current_user_turn=0, reserved_answer=0,
        )

        def counts_exactly_ten(turns):
            return 10

        turns = [Turn(role="user", content="x")]
        assert history_exceeds_budget(turns, budget, token_counter=counts_exactly_ten) is False

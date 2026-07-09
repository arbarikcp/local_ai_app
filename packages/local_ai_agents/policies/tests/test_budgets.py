import pytest

from local_ai_agents.policies.budgets import ToolBudget
from local_ai_agents.tools.base import ToolBudgetExceededError


class TestConsume:
    def test_consuming_within_budget_succeeds(self):
        budget = ToolBudget(max_total_calls=3)
        budget.consume("calculator")
        assert budget.calls_for("calculator") == 1

    def test_raises_once_the_total_budget_is_exhausted(self):
        budget = ToolBudget(max_total_calls=2)
        budget.consume("calculator")
        budget.consume("calculator")
        with pytest.raises(ToolBudgetExceededError):
            budget.consume("calculator")

    def test_raises_once_the_per_tool_budget_is_exhausted(self):
        budget = ToolBudget(max_total_calls=10, max_calls_per_tool=1)
        budget.consume("calculator")
        with pytest.raises(ToolBudgetExceededError):
            budget.consume("calculator")

    def test_per_tool_budget_does_not_affect_other_tools(self):
        budget = ToolBudget(max_total_calls=10, max_calls_per_tool=1)
        budget.consume("calculator")
        budget.consume("search_files")  # different tool, its own counter
        assert budget.calls_for("search_files") == 1

    def test_no_per_tool_limit_by_default(self):
        budget = ToolBudget(max_total_calls=10)
        budget.consume("calculator")
        budget.consume("calculator")
        assert budget.calls_for("calculator") == 2


class TestRemainingTotalCalls:
    def test_decreases_as_calls_are_consumed(self):
        budget = ToolBudget(max_total_calls=5)
        budget.consume("calculator")
        assert budget.remaining_total_calls == 4

    def test_never_goes_negative(self):
        budget = ToolBudget(max_total_calls=1)
        budget.consume("calculator")
        with pytest.raises(ToolBudgetExceededError):
            budget.consume("calculator")
        assert budget.remaining_total_calls == 0

import time

import pytest

from local_ai_agents.planners.safety_budget import AgentSafetyBudget, SafetyBudgetExceededError


def make_budget(**overrides) -> AgentSafetyBudget:
    defaults = dict(max_steps=8, max_tool_calls=5, max_runtime_seconds=60, max_tokens_total=8000)
    defaults.update(overrides)
    return AgentSafetyBudget(**defaults)


class TestRecordStep:
    def test_steps_within_budget_succeed(self):
        budget = make_budget(max_steps=2)
        budget.record_step()
        budget.record_step()
        assert budget.steps_taken == 2

    def test_raises_once_max_steps_is_exceeded(self):
        budget = make_budget(max_steps=1)
        budget.record_step()
        with pytest.raises(SafetyBudgetExceededError):
            budget.record_step()


class TestRecordToolCall:
    def test_raises_once_max_tool_calls_is_exceeded(self):
        budget = make_budget(max_tool_calls=1)
        budget.record_tool_call()
        with pytest.raises(SafetyBudgetExceededError):
            budget.record_tool_call()


class TestRecordTokens:
    def test_accumulates_across_calls(self):
        budget = make_budget(max_tokens_total=1000)
        budget.record_tokens(400)
        budget.record_tokens(400)
        assert budget.tokens_used == 800

    def test_raises_once_total_tokens_exceed_the_budget(self):
        budget = make_budget(max_tokens_total=500)
        budget.record_tokens(400)
        with pytest.raises(SafetyBudgetExceededError):
            budget.record_tokens(200)


class TestCheckRuntime:
    def test_does_not_raise_within_budget(self):
        budget = make_budget(max_runtime_seconds=60)
        budget.check_runtime()

    def test_raises_once_wall_clock_time_is_exceeded(self):
        budget = make_budget(max_runtime_seconds=0.01)
        time.sleep(0.02)
        with pytest.raises(SafetyBudgetExceededError):
            budget.check_runtime()


class TestRequiresApproval:
    def test_true_for_a_listed_tool(self):
        budget = make_budget()
        budget.requires_human_approval = ["file_write", "shell_exec"]
        assert budget.requires_approval("file_write") is True

    def test_false_for_an_unlisted_tool(self):
        budget = make_budget()
        budget.requires_human_approval = ["file_write"]
        assert budget.requires_approval("calculator") is False

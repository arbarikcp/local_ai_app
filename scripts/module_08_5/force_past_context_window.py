"""Lab 3 — force a conversation past the context window and observe
truncation actually engaging.

Needs NO live model runtime: this exercises our own budget/truncation
logic against a synthetic conversation, not a model's generation quality -
genuinely runnable and provable right now.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "packages"))

from local_ai_core.conversation.token_budget import (  # noqa: E402
    ConversationBudget,
    heuristic_token_counter,
    history_exceeds_budget,
)
from local_ai_core.conversation.truncation import drop_oldest  # noqa: E402
from local_ai_core.conversation.turn import Turn  # noqa: E402


def build_synthetic_conversation(n_turns: int, words_per_turn: int = 50) -> list[Turn]:
    turns = [Turn(role="system", content="You are a helpful assistant.", sticky=True)]
    for i in range(n_turns):
        role = "user" if i % 2 == 0 else "assistant"
        content = f"Turn {i}: " + " ".join(["word"] * words_per_turn)
        turns.append(Turn(role=role, content=content))
    return turns


def run_lab(n_turns: int = 40, context_window: int = 2000) -> dict:
    turns = build_synthetic_conversation(n_turns)
    budget = ConversationBudget(
        context_window=context_window, reserved_system=100, reserved_tools=0,
        reserved_current_user_turn=100, reserved_answer=300,
    )

    exceeded_before = history_exceeds_budget(turns, budget, heuristic_token_counter)
    truncated = drop_oldest(turns, budget, heuristic_token_counter)
    exceeded_after = history_exceeds_budget(truncated, budget, heuristic_token_counter)

    return {
        "original_turn_count": len(turns),
        "original_token_estimate": heuristic_token_counter(turns),
        "history_budget": budget.history_budget,
        "exceeded_before_truncation": exceeded_before,
        "truncated_turn_count": len(truncated),
        "truncated_token_estimate": heuristic_token_counter(truncated),
        "exceeded_after_truncation": exceeded_after,
        "system_prompt_retained": any(t.sticky for t in truncated),
    }


def result_to_markdown(result: dict) -> str:
    return (
        "# Lab 3 — forcing a conversation past the context window\n\n"
        f"- Original turns: {result['original_turn_count']} (~{result['original_token_estimate']} tokens)\n"
        f"- History budget: {result['history_budget']} tokens\n"
        f"- Exceeded budget before truncation: {result['exceeded_before_truncation']}\n"
        f"- After drop_oldest: {result['truncated_turn_count']} turns "
        f"(~{result['truncated_token_estimate']} tokens)\n"
        f"- Exceeded budget after truncation: {result['exceeded_after_truncation']}\n"
        f"- Sticky system prompt retained: {result['system_prompt_retained']}\n"
    )


def main() -> int:
    print(result_to_markdown(run_lab()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

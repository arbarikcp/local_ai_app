"""Lab 4 — compare drop-oldest vs. summarization buffer on the same
conversation: which strategy retains an early fact better?

Needs NO live model runtime for the comparison mechanics - a deterministic
crude "summarizer" stands in for a real LLMRuntime.generate() call so the
lab is provable right now; real usage would inject a real summarize_fn.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "packages"))

from local_ai_core.conversation.summarizer import summarize_then_truncate  # noqa: E402
from local_ai_core.conversation.token_budget import ConversationBudget, heuristic_token_counter  # noqa: E402
from local_ai_core.conversation.truncation import drop_oldest  # noqa: E402
from local_ai_core.conversation.turn import Turn  # noqa: E402

EARLY_FACT = "My account number is ACC-88213."
EARLY_FACT_MARKER = "ACC-88213"


def build_conversation_with_early_fact(n_turns: int, early_fact: str = EARLY_FACT) -> list[Turn]:
    turns = [
        Turn(role="system", content="You are a helpful assistant.", sticky=True),
        Turn(role="user", content=early_fact),
        Turn(role="assistant", content="Got it, I'll remember that."),
    ]
    for i in range(n_turns):
        role = "user" if i % 2 == 0 else "assistant"
        turns.append(Turn(role=role, content=f"Turn {i}: " + " ".join(["filler"] * 40)))
    return turns


def fact_appears(turns: list[Turn], marker: str) -> bool:
    return any(marker in t.content for t in turns)


def crude_summarize_fn(turns_to_summarize: list[Turn]) -> str:
    """A simple, deterministic stand-in summarizer: the first sentence of
    each turn, joined. Real usage would call an LLMRuntime; this keeps the
    lab runnable without one while still doing genuine (if crude)
    compression, not a hand-picked answer that happens to keep the fact.
    """
    pieces = [t.content.split(".")[0] for t in turns_to_summarize if t.content]
    return " / ".join(pieces)[:300]


def run_lab(n_turns: int = 30, context_window: int = 1500) -> dict:
    turns = build_conversation_with_early_fact(n_turns)
    budget = ConversationBudget(
        context_window=context_window, reserved_system=100, reserved_tools=0,
        reserved_current_user_turn=100, reserved_answer=300,
    )

    dropped = drop_oldest(turns, budget, heuristic_token_counter)
    summarized = summarize_then_truncate(turns, budget, crude_summarize_fn, heuristic_token_counter, keep_last_n_raw=2)

    return {
        "original_turn_count": len(turns),
        "drop_oldest_turn_count": len(dropped),
        "drop_oldest_fact_present": fact_appears(dropped, EARLY_FACT_MARKER),
        "summarize_turn_count": len(summarized),
        "summarize_fact_present": fact_appears(summarized, EARLY_FACT_MARKER),
    }


def result_to_markdown(result: dict) -> str:
    return (
        "# Lab 4 — drop-oldest vs. summarization buffer\n\n"
        f"Early fact injected: `{EARLY_FACT}`\n\n"
        f"| Strategy | Final turn count | Early fact recoverable |\n|---|---:|---:|\n"
        f"| drop_oldest | {result['drop_oldest_turn_count']} | "
        f"{'yes' if result['drop_oldest_fact_present'] else 'no'} |\n"
        f"| summarize_then_truncate | {result['summarize_turn_count']} | "
        f"{'yes' if result['summarize_fact_present'] else 'no'} |\n"
    )


def main() -> int:
    print(result_to_markdown(run_lab()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

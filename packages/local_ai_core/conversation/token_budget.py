"""ConversationBudget and token-aware history accounting (theory doc §2,
"Conversation budget").

Every function here accepts an injected token counter rather than counting
words itself - real usage plugs in a tokenizer-backed counter that renders
the actual chat template first (Module 1's HFTokenizerCounter.count_chat,
Module 6's MLXRuntime.render_prompt do this); this module's own default is
an explicitly-labeled heuristic, never trusted for a real budget decision -
the same discipline Module 1 established for token counting generally.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .turn import Turn, to_chat_messages

TokenCounterFn = Callable[[list[Turn]], int]


@dataclass(frozen=True)
class ConversationBudget:
    context_window: int
    reserved_system: int
    reserved_tools: int
    reserved_current_user_turn: int
    reserved_answer: int

    def __post_init__(self) -> None:
        for name in ("context_window", "reserved_system", "reserved_tools", "reserved_current_user_turn", "reserved_answer"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")

    @property
    def history_budget(self) -> int:
        """Never negative - a misconfigured/over-reserved budget reads as
        "no room for history," not a silent underflow.
        """
        remaining = (
            self.context_window
            - self.reserved_system
            - self.reserved_tools
            - self.reserved_current_user_turn
            - self.reserved_answer
        )
        return max(0, remaining)


# Empirically, English prose across common BPE-style tokenizers averages
# roughly 1.3 tokens per whitespace-split word (same figure and caveat as
# Module 1's token_estimate.py) - a labeled approximation, not a budgeting
# tool a caller should trust for a real decision.
_HEURISTIC_TOKENS_PER_WORD = 1.3


def heuristic_token_counter(turns: list[Turn]) -> int:
    """Rough word-count-based estimate. Real usage should inject a
    tokenizer-backed counter instead (see module docstring).
    """
    messages = to_chat_messages(turns)
    text = " ".join(m["content"] for m in messages)
    words = text.split()
    return max(0, round(len(words) * _HEURISTIC_TOKENS_PER_WORD))


def history_exceeds_budget(
    turns: list[Turn], budget: ConversationBudget, token_counter: TokenCounterFn = heuristic_token_counter
) -> bool:
    return token_counter(turns) > budget.history_budget

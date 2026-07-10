"""Intent classification (ARCHITECTURE.md "Intent classification") —
nothing in the repo classifies a free-text request into a task type
(confirmed by survey); Module 17's own pipeline is single-shaped. Real
keyword/pattern matching, the same discipline Module 18's
`should_use_vlm()` and Module 20's `route_model()` applied to their own
routing decisions: a traceable decision with a stated reason, not a vibe.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class IntentType(Enum):
    EXPLAIN_REPO = "explain_repo"
    SEARCH_CODE = "search_code"
    EXPLAIN_SYMBOL = "explain_symbol"
    GENERATE_TESTS = "generate_tests"
    SUGGEST_REFACTOR = "suggest_refactor"
    PROPOSE_PATCH = "propose_patch"
    RUN_TESTS = "run_tests"


@dataclass(frozen=True)
class IntentClassification:
    intent: IntentType
    reason: str


# Checked in this order - a request naming "test" alongside "generate"/"write"
# is a test-generation request even though it also contains "test"; checked
# before the plain run_tests keywords so "generate a test" doesn't
# misclassify as "run the tests".
_RULES: tuple[tuple[IntentType, tuple[str, ...]], ...] = (
    (IntentType.GENERATE_TESTS, ("generate test", "write test", "add test", "create test")),
    (IntentType.RUN_TESTS, ("run test", "run the test", "execute test")),
    (IntentType.SUGGEST_REFACTOR, ("refactor", "clean up", "improve the code", "simplify")),
    (IntentType.PROPOSE_PATCH, ("fix", "patch", "bug", "propose a change", "change the code")),
    (IntentType.EXPLAIN_SYMBOL, ("explain function", "explain class", "explain method", "what does")),
    (IntentType.SEARCH_CODE, ("search", "find", "where is", "look for")),
    (IntentType.EXPLAIN_REPO, ("explain the repo", "repo structure", "overview", "what is this repo")),
)

_DEFAULT_INTENT = IntentType.EXPLAIN_REPO


def classify_intent(request: str) -> IntentClassification:
    lowered = request.lower()
    for intent, keywords in _RULES:
        for keyword in keywords:
            if keyword in lowered:
                return IntentClassification(intent=intent, reason=f"matched keyword {keyword!r}")
    return IntentClassification(
        intent=_DEFAULT_INTENT, reason="no keyword matched - defaulting to explain_repo"
    )

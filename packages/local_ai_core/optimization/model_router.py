"""Small model router (theory doc §13) — real task-complexity signals mapped
to a model tier, the same discipline Module 19's `recommend_approach()`
applied to the prompting/RAG/fine-tuning decision and Module 18's
`should_use_vlm()` applied to its own routing diagram: a traceable
decision, not a vibe.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ModelTier(Enum):
    SMALL = "small"
    LARGE = "large"


@dataclass(frozen=True)
class RoutingDecision:
    tier: ModelTier
    reason: str


def route_model(
    *,
    prompt_token_count: int,
    requires_multi_step_reasoning: bool = False,
    requires_tool_calls: bool = False,
    output_must_be_structured: bool = False,
    large_model_token_threshold: int = 2000,
) -> RoutingDecision:
    """Escalates to the large model tier on any single strong signal
    (curriculum's "route heavy tasks to larger model only when needed" -
    each signal below is independently a reason heavy enough to justify the
    larger model's cost, so the first one found wins rather than requiring
    all of them at once, the opposite of Module 19's fine-tuning gate).
    Falls back to the small tier only when none of the escalation signals
    are present.
    """
    if requires_multi_step_reasoning:
        return RoutingDecision(
            tier=ModelTier.LARGE,
            reason="task requires multi-step reasoning - the small model tier is not reliable for this",
        )
    if requires_tool_calls:
        return RoutingDecision(
            tier=ModelTier.LARGE,
            reason="task requires tool calls - the large model tier has more reliable tool-call formatting",
        )
    if output_must_be_structured:
        return RoutingDecision(
            tier=ModelTier.LARGE,
            reason="output must be structured - the large model tier has more reliable schema adherence",
        )
    if prompt_token_count > large_model_token_threshold:
        return RoutingDecision(
            tier=ModelTier.LARGE,
            reason=f"prompt token count ({prompt_token_count}) exceeds the small-model threshold "
            f"({large_model_token_threshold})",
        )
    return RoutingDecision(
        tier=ModelTier.SMALL,
        reason="no escalation signal present - the small model tier is cheaper and sufficient",
    )

"""The prompting-vs-RAG-vs-fine-tuning decision framework (theory doc §1)
- curriculum's own bullet points as real boolean inputs, returning a real
`Recommendation` with the chosen approach and the specific reason - a
traceable decision, not a vibe. Same discipline Module 18's
`should_use_vlm()` applied to its own decision diagram.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Approach(Enum):
    PROMPTING = "prompting"
    RAG = "rag"
    FINE_TUNING = "fine_tuning"


@dataclass(frozen=True)
class Recommendation:
    approach: Approach
    reason: str


def recommend_approach(
    *,
    task_is_simple: bool = False,
    behavior_specifiable_in_instructions: bool = False,
    needs_private_or_changing_knowledge: bool = False,
    must_cite_sources: bool = False,
    output_style_is_repetitive: bool = False,
    task_is_narrow_and_stable: bool = False,
    has_enough_labeled_data: bool = False,
    evaluation_proves_improvement: bool = False,
) -> Recommendation:
    """Checked in curriculum's own stated priority: never fine-tune just
    to add knowledge that changes often (explicitly checked first, ahead
    of every fine-tuning signal, so a "yes" here can never be overridden
    by unrelated fine-tuning signals) - RAG and prompting are cheaper to
    build, run, and keep correct than a fine-tune, so fine-tuning only
    wins when its own preconditions are *all* true, not just some.
    """
    if needs_private_or_changing_knowledge or must_cite_sources:
        return Recommendation(
            approach=Approach.RAG,
            reason="knowledge is private/changing or answers must cite sources - do not fine-tune to add facts that change; use RAG",
        )

    fine_tuning_ready = (
        output_style_is_repetitive
        and task_is_narrow_and_stable
        and has_enough_labeled_data
        and evaluation_proves_improvement
    )
    if fine_tuning_ready:
        return Recommendation(
            approach=Approach.FINE_TUNING,
            reason="output style is repetitive, the task is narrow and stable, enough labeled data exists, and evaluation proves improvement",
        )

    if task_is_simple or behavior_specifiable_in_instructions:
        return Recommendation(
            approach=Approach.PROMPTING,
            reason="task is simple and/or behavior can be fully specified in instructions",
        )

    return Recommendation(
        approach=Approach.PROMPTING,
        reason="no strong signal for RAG or fine-tuning - start with prompting, the cheapest approach to try first",
    )

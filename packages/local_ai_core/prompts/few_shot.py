"""Few-shot and negative examples (theory doc §4-5).

Positive examples teach format compliance more reliably than prose alone;
negative examples target specific, predictable failure modes (wrapping
JSON in markdown fences, inventing a missing field) rather than teaching
the task in general. Both are distinct tools, not interchangeable.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FewShotExample:
    """A correct input -> output pair, shown to teach format compliance."""

    input: str
    output: str


@dataclass(frozen=True)
class NegativeExample:
    """An incorrect output for a given input, with the reason it's wrong.

    Targets a specific, known failure mode rather than teaching the task.
    """

    input: str
    wrong_output: str
    reason: str


def format_few_shot_examples(examples: list[FewShotExample]) -> str:
    if not examples:
        return ""
    blocks = [f"Input: {ex.input}\nOutput: {ex.output}" for ex in examples]
    return "\n\n".join(blocks)


def format_negative_examples(examples: list[NegativeExample]) -> str:
    if not examples:
        return ""
    blocks = [
        f"Input: {ex.input}\nDo NOT output: {ex.wrong_output}\nWhy: {ex.reason}" for ex in examples
    ]
    return "\n\n".join(blocks)

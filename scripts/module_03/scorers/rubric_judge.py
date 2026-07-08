"""LLM-as-judge rubric scoring.

Inherits the judge model's own biases and blind spots (docs/modules/03_local
_model_selection_and_benchmarking.md §9 — "the judge-model problem",
revisited in depth in Module 13). Treat rubric scores as a fast, repeatable
*signal*, not ground truth — calibrate periodically against human review.

The judge call is injected as a plain ``Callable[[str], str]`` (prompt in,
raw response text out) rather than depending on any runtime abstraction —
Module 3 is pre-abstraction (that's Module 6's job), and injecting the judge
function is what makes this module testable without a live model: tests
pass a fake judge, real usage passes a function backed by
``ollama_probe.generate`` (Module 1) or whichever runtime is configured.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

JudgeFn = Callable[[str], str]

DEFAULT_RUBRIC_TEMPLATE = """\
You are grading an AI assistant's response for the task below. Score it from 1 (poor) to 5 \
(excellent) on how well it accomplishes the task. Respond with ONLY a line of the form \
"Score: <1-5>" followed by one short sentence of justification.

Task:
{task_description}

Reference answer (if available):
{reference}

Model's response to grade:
{prediction}
"""

_SCORE_RE = re.compile(r"score\s*:?\s*([1-5])", re.IGNORECASE)


@dataclass(frozen=True)
class JudgeResult:
    score: int | None
    raw_response: str
    parse_succeeded: bool


def build_judge_prompt(
    task_description: str,
    prediction: str,
    reference: str | None = None,
    template: str = DEFAULT_RUBRIC_TEMPLATE,
) -> str:
    return template.format(
        task_description=task_description,
        reference=reference or "(no reference provided)",
        prediction=prediction,
    )


def parse_judge_score(raw_response: str) -> int | None:
    """Extract a 1-5 integer score from a judge's free-text response.

    Returns None when no score pattern is found — callers must treat that as
    a failed judgment, not a score of 0, since 0 is not on the rubric scale.
    """
    match = _SCORE_RE.search(raw_response)
    if match is None:
        return None
    return int(match.group(1))


def judge(
    judge_fn: JudgeFn,
    task_description: str,
    prediction: str,
    reference: str | None = None,
    template: str = DEFAULT_RUBRIC_TEMPLATE,
) -> JudgeResult:
    prompt = build_judge_prompt(task_description, prediction, reference, template)
    raw_response = judge_fn(prompt)
    score = parse_judge_score(raw_response)
    return JudgeResult(score=score, raw_response=raw_response, parse_succeeded=score is not None)


def mean_score(results: list[JudgeResult]) -> float | None:
    """Mean of successfully-parsed scores; None if none parsed."""
    parsed = [r.score for r in results if r.score is not None]
    if not parsed:
        return None
    return sum(parsed) / len(parsed)

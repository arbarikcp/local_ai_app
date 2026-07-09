"""Before/after fine-tuning evaluation harness (theory doc §9). Reuses
Module 13's `answer_metrics.must_contain_score` against a golden set,
scoring a baseline runtime and a "fine-tuned" runtime side by side with
real aggregate deltas.

Since no model is actually fine-tuned on this machine, both sides are
`FakeRuntime`-backed in tests with deliberately different scripted quality -
honest about being a harness demonstration, not a real fine-tuning result
(theory doc's Machine note).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from local_ai_core.evals.answer_metrics import must_contain_score
from local_ai_core.runtimes.types import LLMRequest


class GeneratingRuntime(Protocol):
    async def generate(self, request: LLMRequest) -> object: ...


@dataclass(frozen=True)
class GoldenCase:
    question: str
    must_contain: list[str]


@dataclass(frozen=True)
class CaseScore:
    question: str
    baseline_score: float
    candidate_score: float

    @property
    def delta(self) -> float:
        return self.candidate_score - self.baseline_score


@dataclass(frozen=True)
class ComparisonReport:
    case_scores: list[CaseScore]
    baseline_mean_score: float
    candidate_mean_score: float

    @property
    def mean_delta(self) -> float:
        return self.candidate_mean_score - self.baseline_mean_score

    @property
    def candidate_improved(self) -> bool:
        return self.mean_delta > 0


async def _score_runtime(runtime: GeneratingRuntime, model: str, cases: list[GoldenCase]) -> list[float]:
    scores = []
    for case in cases:
        request = LLMRequest(model=model, prompt=case.question)
        response = await runtime.generate(request)
        scores.append(must_contain_score(response.text, case.must_contain))
    return scores


async def compare_before_after(
    cases: list[GoldenCase],
    *,
    baseline_runtime: GeneratingRuntime,
    candidate_runtime: GeneratingRuntime,
    model: str,
) -> ComparisonReport:
    """Runs the same golden set against a baseline runtime (e.g. prompt-only)
    and a candidate runtime (e.g. "fine-tuned"), scoring each with
    `must_contain_score` and returning per-case and aggregate deltas.
    """
    if not cases:
        raise ValueError("cases must not be empty")

    baseline_scores = await _score_runtime(baseline_runtime, model, cases)
    candidate_scores = await _score_runtime(candidate_runtime, model, cases)

    case_scores = [
        CaseScore(question=case.question, baseline_score=baseline_score, candidate_score=candidate_score)
        for case, baseline_score, candidate_score in zip(cases, baseline_scores, candidate_scores)
    ]

    return ComparisonReport(
        case_scores=case_scores,
        baseline_mean_score=sum(baseline_scores) / len(baseline_scores),
        candidate_mean_score=sum(candidate_scores) / len(candidate_scores),
    )

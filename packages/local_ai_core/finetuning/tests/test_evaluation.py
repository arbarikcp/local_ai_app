import pytest

from local_ai_core.finetuning.evaluation import GoldenCase, compare_before_after

MODEL = "test-model"


@pytest.fixture
def golden_cases() -> list[GoldenCase]:
    return [
        GoldenCase(question="What category is a double-charge ticket?", must_contain=["billing"]),
        GoldenCase(question="What category is a 2FA backup code ticket?", must_contain=["security"]),
    ]


class TestCompareBeforeAfter:
    async def test_candidate_that_answers_correctly_scores_higher_than_a_weak_baseline(self, golden_cases):
        baseline = _ScriptedRuntime(["I'm not sure.", "I'm not sure."])
        candidate = _ScriptedRuntime(["billing", "security"])

        report = await compare_before_after(
            golden_cases, baseline_runtime=baseline, candidate_runtime=candidate, model=MODEL
        )

        assert report.candidate_mean_score == 1.0
        assert report.baseline_mean_score == 0.0
        assert report.candidate_improved is True
        assert report.mean_delta == 1.0

    async def test_identical_runtimes_produce_zero_delta(self, golden_cases):
        runtime_a = _ScriptedRuntime(["billing", "security"])
        runtime_b = _ScriptedRuntime(["billing", "security"])

        report = await compare_before_after(
            golden_cases, baseline_runtime=runtime_a, candidate_runtime=runtime_b, model=MODEL
        )

        assert report.mean_delta == 0.0
        assert report.candidate_improved is False

    async def test_per_case_scores_are_reported_individually(self, golden_cases):
        baseline = _ScriptedRuntime(["billing", "wrong answer"])
        candidate = _ScriptedRuntime(["billing", "security"])

        report = await compare_before_after(
            golden_cases, baseline_runtime=baseline, candidate_runtime=candidate, model=MODEL
        )

        assert len(report.case_scores) == 2
        assert report.case_scores[0].delta == 0.0
        assert report.case_scores[1].delta == 1.0

    async def test_empty_golden_set_raises(self):
        runtime = _ScriptedRuntime([])
        with pytest.raises(ValueError):
            await compare_before_after([], baseline_runtime=runtime, candidate_runtime=runtime, model=MODEL)


class _ScriptedRuntime:
    """A minimal fake that returns a different scripted answer per call, in
    order - used where FakeRuntime's per-model (not per-call) response map
    isn't granular enough to script per-question answers.
    """

    def __init__(self, answers: list[str]) -> None:
        self._answers = list(answers)
        self._call_index = 0

    async def generate(self, request):
        from local_ai_core.runtimes.types import LLMResponse

        answer = self._answers[self._call_index]
        self._call_index += 1
        return LLMResponse(text=answer, model=request.model)

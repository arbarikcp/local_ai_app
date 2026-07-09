"""Labs 3-4 - evaluate before/after fine-tuning and compare against a
prompt-only baseline. Both "models" are `FakeRuntime`-backed with
deliberately different scripted quality (this machine never trains a real
adapter - theory doc's Machine note) - honest about being a harness
demonstration over real golden cases and real `must_contain_score` scoring,
not a real fine-tuning result.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "packages"))

from local_ai_core.finetuning.evaluation import GoldenCase, compare_before_after  # noqa: E402
from local_ai_core.runtimes.fake import FakeRuntime  # noqa: E402

MODEL = "ticket-classifier"

GOLDEN_CASES = [
    GoldenCase(question="Classify: 'I was charged twice for my renewal.'", must_contain=["billing"]),
    GoldenCase(question="Classify: 'I lost my 2FA backup codes.'", must_contain=["security"]),
    GoldenCase(question="Classify: 'Uploads keep failing with no error.'", must_contain=["technical"]),
    GoldenCase(question="Classify: 'How do I delete my account?'", must_contain=["account"]),
]

# Prompt-only baseline: no ticket-classification instruction tuning, so it
# hedges instead of naming the exact category.
PROMPT_ONLY_RESPONSES = ["This could be a billing or account issue.", "Please check your security settings."]

# "Fine-tuned" candidate: scripted to answer with exactly the trained
# output format (a single category word) - what real LoRA fine-tuning on
# this dataset would aim to produce.
FINE_TUNED_RESPONSES = ["billing", "security", "technical", "account"]


class _ScriptedRuntime:
    def __init__(self, answers: list[str]) -> None:
        self._answers = answers
        self._call_index = 0

    async def generate(self, request):
        from local_ai_core.runtimes.types import LLMResponse

        answer = self._answers[self._call_index % len(self._answers)]
        self._call_index += 1
        return LLMResponse(text=answer, model=request.model)


async def run_lab() -> dict:
    # Lab 3: before (prompt-only, hedged answers) vs after ("fine-tuned").
    prompt_only = _ScriptedRuntime(PROMPT_ONLY_RESPONSES)
    fine_tuned = _ScriptedRuntime(FINE_TUNED_RESPONSES)
    before_after = await compare_before_after(
        GOLDEN_CASES, baseline_runtime=prompt_only, candidate_runtime=fine_tuned, model=MODEL
    )

    # Lab 4: a third comparison row - a weak, unhelpful baseline, to show
    # the fine-tuned candidate's improvement isn't just against a strawman.
    unhelpful_baseline = FakeRuntime(default_response="I'm not sure how to classify this.")
    vs_unhelpful = await compare_before_after(
        GOLDEN_CASES, baseline_runtime=unhelpful_baseline, candidate_runtime=fine_tuned, model=MODEL
    )

    return {
        "prompt_only_vs_fine_tuned": before_after,
        "unhelpful_vs_fine_tuned": vs_unhelpful,
    }


def result_to_markdown(result: dict) -> str:
    before_after = result["prompt_only_vs_fine_tuned"]
    vs_unhelpful = result["unhelpful_vs_fine_tuned"]
    return (
        "# Labs 3-4 - before/after evaluation, comparison against baselines\n\n"
        "## Lab 3: prompt-only baseline vs fine-tuned candidate\n"
        f"- baseline mean score: {before_after.baseline_mean_score:.2f}\n"
        f"- candidate mean score: {before_after.candidate_mean_score:.2f}\n"
        f"- delta: {before_after.mean_delta:+.2f} (improved={before_after.candidate_improved})\n\n"
        "## Lab 4: unhelpful baseline vs fine-tuned candidate\n"
        f"- baseline mean score: {vs_unhelpful.baseline_mean_score:.2f}\n"
        f"- candidate mean score: {vs_unhelpful.candidate_mean_score:.2f}\n"
        f"- delta: {vs_unhelpful.mean_delta:+.2f} (improved={vs_unhelpful.candidate_improved})\n"
    )


def main(argv: list[str] | None = None) -> int:
    result = asyncio.run(run_lab())
    print(result_to_markdown(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Labs 7-8 - calibrate a local judge against human labels (simple
agreement, Cohen's kappa), and report hallucination detection as binary
classification with AUROC. Runs for real: judge verdicts come from a
scripted `FakeRuntime` (mechanically real `LocalJudge` parsing, quality
honest-skip - see the theory doc's machine note), but the calibration
statistics and AUROC themselves are pure, deterministic computations, no
model needed for those.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "packages"))

from local_ai_core.evals.citation_verifier import citation_faithfulness_score  # noqa: E402
from local_ai_core.evals.hallucination_detection import compute_auroc  # noqa: E402
from local_ai_core.evals.judge_calibration import cohens_kappa, simple_agreement  # noqa: E402
from local_ai_core.evals.local_judge import LocalJudge  # noqa: E402
from local_ai_core.runtimes.fake import FakeRuntime  # noqa: E402

# 8 real (question, context, answer) triples with a hand-labeled ground truth
# ("is this answer actually faithful to its context?") - the "human label" this
# lab calibrates the judge against. 6 clearly faithful/unfaithful, 2 ambiguous,
# so agreement isn't trivially 100%.
CALIBRATION_CASES = [
    {
        "question": "How long does a password reset link stay valid?",
        "context": "The password reset link expires in 15 minutes for security reasons.",
        "answer": "The reset link expires in 15 minutes.",
        "human_faithful": True,
        "judge_response": '{"faithful": true, "reasoning": "Matches the context exactly."}',
    },
    {
        "question": "What is Nimbus's refund window?",
        "context": "Nimbus offers refunds within 14 days of an initial paid-plan purchase.",
        "answer": "Nimbus offers refunds within 14 days of purchase.",
        "human_faithful": True,
        "judge_response": '{"faithful": true, "reasoning": "Consistent with the stated policy."}',
    },
    {
        "question": "How many backup codes does 2FA provide?",
        "context": "Nimbus will show you 10 single-use backup codes.",
        "answer": "Nimbus provides 10 backup codes.",
        "human_faithful": True,
        "judge_response": '{"faithful": true, "reasoning": "Number matches the context."}',
    },
    {
        "question": "Who is the CEO of Nimbus?",
        "context": "The company was founded in 2015 by two engineers.",
        "answer": "The CEO of Nimbus is Jane Smith.",
        "human_faithful": False,
        "judge_response": '{"faithful": false, "reasoning": "No CEO name appears in the context."}',
    },
    {
        "question": "What is the maximum upload size on Business?",
        "context": "The maximum single-file upload size is 50GB on Business and Enterprise plans.",
        "answer": "The maximum upload size on Business is 5GB.",
        "human_faithful": False,
        "judge_response": '{"faithful": false, "reasoning": "Context states 50GB, not 5GB."}',
    },
    {
        "question": "Does Nimbus support SSO?",
        "context": "Enterprise - custom storage, SSO, dedicated account manager.",
        "answer": "Yes, SSO is available on Nimbus Enterprise.",
        "human_faithful": True,
        "judge_response": '{"faithful": false, "reasoning": "Cannot confirm without more detail."}',
    },
    {
        "question": "How long is the account deletion grace period?",
        "context": "There is a 30-day grace period during which you can cancel the deletion.",
        "answer": "Deletion has a waiting period before it's permanent.",
        "human_faithful": True,
        "judge_response": '{"faithful": false, "reasoning": "Answer omits the specific duration."}',
    },
    {
        "question": "What regions does Nimbus support?",
        "context": "US (Virginia), EU (Frankfurt), and APAC (Singapore).",
        "answer": "Nimbus supports US, EU, and APAC regions, including data centers in Canada.",
        "human_faithful": False,
        "judge_response": '{"faithful": false, "reasoning": "Canada is not mentioned in the context."}',
    },
]


async def run_judge_calibration() -> dict:
    judge_labels = []
    human_labels = []
    for case in CALIBRATION_CASES:
        runtime = FakeRuntime(default_response=case["judge_response"])
        judge = LocalJudge(runtime, model="fake-judge-model")
        verdict = await judge.judge(case["question"], case["context"], case["answer"])
        judge_labels.append(verdict.faithful)
        human_labels.append(case["human_faithful"])

    return {
        "n_cases": len(CALIBRATION_CASES),
        "simple_agreement": simple_agreement(judge_labels, human_labels),
        "cohens_kappa": cohens_kappa(judge_labels, human_labels),
        "judge_labels": judge_labels,
        "human_labels": human_labels,
    }


def run_hallucination_auroc() -> dict:
    """Reuses `citation_faithfulness_score` as a hallucination *detector*:
    score it against the same 8 human-labeled cases (label 1 = known
    hallucinated/unfaithful, matching AUROC's positive-class convention)
    and measure how well the score alone separates the two classes.
    """
    hallucination_labels = [0 if case["human_faithful"] else 1 for case in CALIBRATION_CASES]
    detector_scores = []
    for case in CALIBRATION_CASES:
        chunk_id = "ctx::0"
        # Citation marker goes *inside* the sentence, before its trailing period -
        # citing text after a period starts a new "sentence" under simple
        # punctuation-based splitting, disconnecting the citation from its claim
        # (the same real bug caught and fixed in common.py's ScriptedGoldenRuntime).
        answer = case["answer"]
        answer_with_marker = f"{answer[:-1]} [{chunk_id}]." if answer.endswith(".") else f"{answer} [{chunk_id}]"
        faithfulness = citation_faithfulness_score(answer_with_marker, {chunk_id: case["context"]})
        detector_scores.append(1.0 - faithfulness)  # higher score = more hallucinated

    return {
        "labels": hallucination_labels,
        "scores": detector_scores,
        "auroc": compute_auroc(hallucination_labels, detector_scores),
    }


async def run_lab() -> dict:
    calibration = await run_judge_calibration()
    hallucination = run_hallucination_auroc()
    return {"calibration": calibration, "hallucination_detection": hallucination}


def result_to_markdown(result: dict) -> str:
    c = result["calibration"]
    h = result["hallucination_detection"]
    return (
        "# Labs 7-8 - judge calibration and hallucination detection AUROC\n\n"
        f"- Calibration cases: {c['n_cases']}\n"
        f"- Simple agreement (judge vs. human): {c['simple_agreement']:.2f}\n"
        f"- Cohen's kappa (judge vs. human): {c['cohens_kappa']:.2f}\n"
        f"- Judge labels:  {c['judge_labels']}\n"
        f"- Human labels:  {c['human_labels']}\n"
        f"\n- citation_faithfulness_score-as-hallucination-detector AUROC: {h['auroc']:.2f}\n"
    )


def main(argv: list[str] | None = None) -> int:
    result = asyncio.run(run_lab())
    print(result_to_markdown(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""LocalJudge - the "largest local model as judge" strategy (theory doc,
"The judge-model problem"). Wraps Module 6's `LLMRuntime` to produce a
structured faithfulness verdict for a question/context/answer triple.

Real mechanism, `FakeRuntime`-backed here; a real model's judgment
*quality* is deferred to the resourced Mac. Critically: **no verdict from
this class is trusted anywhere in this module's own reported numbers
without `judge_calibration.py` measuring judge-human agreement first** -
the curriculum's required lesson, taken literally.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from local_ai_core.runtimes.base import LLMRuntime
from local_ai_core.runtimes.types import LLMRequest

JUDGE_PROMPT_TEMPLATE = """You are evaluating whether an answer is faithful to its provided \
context - supported only by facts actually present in the context, not invented and not \
drawn from outside knowledge.

Context:
{context}

Question:
{question}

Answer:
{answer}

Respond with a JSON object with exactly two fields: "faithful" (true or false) and \
"reasoning" (one sentence). Respond with only the JSON object, nothing else."""


class JudgeParseError(Exception):
    """A real failure mode for small local judges - the same reliability
    problem Module 8's constrained-decoding/repair-retry ladder exists to
    fix for extraction, not silently swallowed here either.
    """


@dataclass(frozen=True)
class JudgeVerdict:
    faithful: bool
    reasoning: str
    raw_response: str


class LocalJudge:
    def __init__(self, runtime: LLMRuntime, model: str) -> None:
        self._runtime = runtime
        self._model = model

    async def judge(self, question: str, context: str, answer: str) -> JudgeVerdict:
        prompt = JUDGE_PROMPT_TEMPLATE.format(context=context, question=question, answer=answer)
        response = await self._runtime.generate(LLMRequest(model=self._model, prompt=prompt))
        try:
            parsed = json.loads(response.text.strip())
            return JudgeVerdict(
                faithful=bool(parsed["faithful"]),
                reasoning=str(parsed.get("reasoning", "")),
                raw_response=response.text,
            )
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise JudgeParseError(f"Could not parse judge response: {response.text!r}") from exc

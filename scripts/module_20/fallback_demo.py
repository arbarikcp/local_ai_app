"""Lab 4 - add a fallback model. A primary runtime that's down (real
`RuntimeUnavailable`) falls through to a healthy secondary runtime; a
separate scenario shows a non-retryable error propagating immediately
instead of wasting a fallback attempt.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "packages"))

from local_ai_core.optimization.fallback import FallbackRuntime, NoRuntimesAvailable  # noqa: E402
from local_ai_core.runtimes.errors import RuntimeUnavailable, SchemaValidationError  # noqa: E402
from local_ai_core.runtimes.fake import FakeRuntime  # noqa: E402
from local_ai_core.runtimes.types import LLMRequest  # noqa: E402

REQUEST = LLMRequest(model="ticket-classifier", prompt="Classify this ticket.")


async def run_lab() -> dict:
    primary_down = FakeRuntime(fail_with=RuntimeUnavailable("primary runtime is unreachable"))
    secondary_healthy = FakeRuntime(default_response="billing (from fallback runtime)")
    chain = FallbackRuntime([primary_down, secondary_healthy])
    fallback_result = await chain.generate(REQUEST)

    validation_failing = FakeRuntime(fail_with=SchemaValidationError("bad schema"))
    never_called = FakeRuntime(default_response="should never run")
    validation_chain = FallbackRuntime([validation_failing, never_called])
    try:
        await validation_chain.generate(REQUEST)
        validation_propagated = False
    except SchemaValidationError:
        validation_propagated = True

    all_down_chain = FallbackRuntime(
        [
            FakeRuntime(fail_with=RuntimeUnavailable("down")),
            FakeRuntime(fail_with=RuntimeUnavailable("also down")),
        ]
    )
    try:
        await all_down_chain.generate(REQUEST)
        all_failed = False
    except NoRuntimesAvailable:
        all_failed = True

    return {
        "fallback_response": fallback_result.response.text,
        "fallback_runtime_index": fallback_result.runtime_index,
        "fallback_attempts": fallback_result.attempts,
        "validation_error_propagated_without_fallback": validation_propagated,
        "never_called_call_count": never_called.call_count,
        "all_runtimes_down_raised": all_failed,
    }


def result_to_markdown(result: dict) -> str:
    return (
        "# Lab 4 - fallback model\n\n"
        f"- Fell back to runtime index {result['fallback_runtime_index']} after "
        f"{result['fallback_attempts']} attempt(s): \"{result['fallback_response']}\"\n"
        f"- Non-retryable validation error propagated without fallback: "
        f"{result['validation_error_propagated_without_fallback']} "
        f"(secondary runtime call count: {result['never_called_call_count']})\n"
        f"- Every runtime down raised NoRuntimesAvailable: {result['all_runtimes_down_raised']}\n"
    )


def main(argv: list[str] | None = None) -> int:
    result = asyncio.run(run_lab())
    print(result_to_markdown(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

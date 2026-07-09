"""Lab 3 - add a model router. Routes a set of realistic requests to the
small or large model tier via `route_model()`, and drives two `FakeRuntime`
instances accordingly - the routing decision genuinely determines which
runtime is called.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "packages"))

from local_ai_core.optimization.model_router import ModelTier, route_model  # noqa: E402
from local_ai_core.runtimes.fake import FakeRuntime  # noqa: E402
from local_ai_core.runtimes.types import LLMRequest  # noqa: E402

REQUESTS = [
    {"label": "simple classification", "prompt_token_count": 40},
    {"label": "long document summary", "prompt_token_count": 3500},
    {"label": "multi-step agent plan", "prompt_token_count": 100, "requires_multi_step_reasoning": True},
    {"label": "tool-calling request", "prompt_token_count": 80, "requires_tool_calls": True},
]


async def run_lab() -> list[dict]:
    small_runtime = FakeRuntime(default_response="(small model) done")
    large_runtime = FakeRuntime(default_response="(large model) done")
    runtimes = {ModelTier.SMALL: small_runtime, ModelTier.LARGE: large_runtime}

    results = []
    for spec in REQUESTS:
        spec = dict(spec)
        label = spec.pop("label")
        decision = route_model(**spec)
        runtime = runtimes[decision.tier]
        response = await runtime.generate(LLMRequest(model="router-demo", prompt=label))
        results.append(
            {
                "label": label,
                "tier": decision.tier.value,
                "reason": decision.reason,
                "response": response.text,
            }
        )
    return results


def results_to_markdown(results: list[dict]) -> str:
    lines = ["# Lab 3 - model router", ""]
    for r in results:
        lines.append(f"- **{r['label']}** -> {r['tier']} ({r['reason']}) -> {r['response']}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    results = asyncio.run(run_lab())
    print(results_to_markdown(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

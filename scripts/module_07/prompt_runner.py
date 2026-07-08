"""Labs 2-3 — compare prompt variants across models, tracking invalid JSON
rate (reusing Module 3's json_validity scorer rather than re-implementing
it).

Fully runnable against FakeRuntime for infrastructure proof; the real
3-model comparison needs a live runtime and is honest-skip on this machine.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "packages"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "module_01"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "module_03"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from local_ai_core.runtimes.types import LLMRequest  # noqa: E402
from prompt_variants import ALL_VARIANTS, render_variant  # noqa: E402
from scorers.json_validity import is_valid_json  # noqa: E402

DEFAULT_TEST_INPUTS = [
    "Maria moved to Austin last spring. She just turned 29.",
    "The new hire, Priya, starts in the Chicago office next Monday. No age was mentioned.",
    "Contact John immediately.",  # deliberately sparse - no age or city given at all
]


@dataclass(frozen=True)
class VariantResult:
    variant_name: str
    model: str
    outputs: list[str]
    invalid_json_rate: float


async def run_variant_against_model(
    variant_name: str, model: str, runtime, test_inputs: list[str]
) -> VariantResult:
    outputs = []
    for text in test_inputs:
        prompt = render_variant(variant_name, text)
        response = await runtime.generate(LLMRequest(model=model, prompt=prompt))
        outputs.append(response.text)
    invalid_count = sum(1 for o in outputs if not is_valid_json(o))
    return VariantResult(
        variant_name=variant_name,
        model=model,
        outputs=outputs,
        invalid_json_rate=invalid_count / len(outputs) if outputs else 0.0,
    )


async def run_lab(runtime, models: list[str], test_inputs: list[str] | None = None) -> list[VariantResult]:
    test_inputs = test_inputs if test_inputs is not None else DEFAULT_TEST_INPUTS
    results = []
    for variant_name in ALL_VARIANTS:
        for model in models:
            results.append(await run_variant_against_model(variant_name, model, runtime, test_inputs))
    return results


def results_to_markdown_table(results: list[VariantResult]) -> str:
    header = "| Variant | Model | Invalid JSON rate |\n|---|---|---:|\n"
    lines = [f"| {r.variant_name} | {r.model} | {r.invalid_json_rate:.0%} |" for r in results]
    return header + "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    from ollama_probe import is_ollama_available
    from local_ai_core.runtimes.ollama import OllamaRuntime

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", required=True)
    args = parser.parse_args(argv)

    if not is_ollama_available():
        print(
            "SKIPPED: Ollama is not reachable at http://localhost:11434. "
            "Install Ollama (Module 2) and re-run this lab on the resourced 32GB Mac.",
            file=sys.stderr,
        )
        return 1

    async def _run():
        runtime = OllamaRuntime()
        try:
            return await run_lab(runtime, args.models)
        finally:
            await runtime.aclose()

    results = asyncio.run(_run())
    print("# Labs 2-3 — prompt variant comparison\n")
    print(results_to_markdown_table(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

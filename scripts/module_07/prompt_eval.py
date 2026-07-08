"""Lab 5 — prompt regression tests: run frozen test cases against a given
PromptTemplate, asserting properties (schema validity + required keys),
never exact strings (Module 6's Gotcha).

Lab 6 — compress a prompt and compare the quality delta on the same cases,
using prompt_variants.variant_4_with_schema() vs. variant_4_compressed().
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "packages"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "module_01"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "module_03"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from local_ai_core.prompts.template import PromptTemplate  # noqa: E402
from local_ai_core.runtimes.types import LLMRequest  # noqa: E402
from scorers.json_validity import has_required_keys  # noqa: E402

REGRESSION_CASES_PATH = (
    Path(__file__).resolve().parent.parent.parent / "evals" / "prompt_regression" / "extraction_cases.jsonl"
)


def load_regression_cases(path: Path = REGRESSION_CASES_PATH) -> list[dict]:
    cases = []
    with open(path) as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                cases.append(json.loads(stripped))
    return cases


@dataclass(frozen=True)
class RegressionCaseResult:
    case_id: str
    passed: bool
    output: str


async def run_regression_suite(
    template: PromptTemplate, runtime, model: str, cases: list[dict]
) -> list[RegressionCaseResult]:
    results = []
    for case in cases:
        prompt = template.render(case["text"])
        response = await runtime.generate(LLMRequest(model=model, prompt=prompt))
        passed = has_required_keys(response.text, case["schema_keys"])
        results.append(RegressionCaseResult(case_id=case["id"], passed=passed, output=response.text))
    return results


def pass_rate(results: list[RegressionCaseResult]) -> float:
    if not results:
        return 0.0
    return sum(1 for r in results if r.passed) / len(results)


def failing_case_ids(results: list[RegressionCaseResult]) -> list[str]:
    return [r.case_id for r in results if not r.passed]


async def compare_compression(
    full_template: PromptTemplate, compressed_template: PromptTemplate, runtime, model: str, cases: list[dict]
) -> dict:
    full_results = await run_regression_suite(full_template, runtime, model, cases)
    compressed_results = await run_regression_suite(compressed_template, runtime, model, cases)
    return {
        "full_pass_rate": pass_rate(full_results),
        "compressed_pass_rate": pass_rate(compressed_results),
        "full_prompt_chars": len(full_template.invariant_prefix()),
        "compressed_prompt_chars": len(compressed_template.invariant_prefix()),
        "compressed_failing_cases": failing_case_ids(compressed_results),
    }


def compression_results_to_markdown(results: dict) -> str:
    char_reduction = 1 - (results["compressed_prompt_chars"] / results["full_prompt_chars"])
    return (
        "# Lab 6 — prompt compression comparison\n\n"
        f"- Full prompt: {results['full_prompt_chars']} chars, "
        f"{results['full_pass_rate']:.0%} pass rate\n"
        f"- Compressed prompt: {results['compressed_prompt_chars']} chars "
        f"({char_reduction:.0%} shorter), {results['compressed_pass_rate']:.0%} pass rate\n"
        f"- Cases that newly fail under compression: {results['compressed_failing_cases'] or 'none'}\n"
    )


def main(argv: list[str] | None = None) -> int:
    from ollama_probe import is_ollama_available
    from local_ai_core.runtimes.ollama import OllamaRuntime
    import prompt_variants as pv

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen2.5:1.5b")
    args = parser.parse_args(argv)

    if not is_ollama_available():
        print(
            "SKIPPED: Ollama is not reachable at http://localhost:11434. "
            "Install Ollama (Module 2) and re-run this lab on the resourced 32GB Mac.",
            file=sys.stderr,
        )
        return 1

    cases = load_regression_cases()

    async def _run():
        runtime = OllamaRuntime()
        try:
            regression = await run_regression_suite(pv.variant_5_with_few_shot(), runtime, args.model, cases)
            compression = await compare_compression(
                pv.variant_4_with_schema(), pv.variant_4_compressed(), runtime, args.model, cases
            )
            return regression, compression
        finally:
            await runtime.aclose()

    regression, compression = asyncio.run(_run())

    print(f"# Lab 5 — regression suite\n\nPass rate: {pass_rate(regression):.0%}")
    failing = failing_case_ids(regression)
    if failing:
        print(f"Failing cases: {failing}")
    print()
    print(compression_results_to_markdown(compression))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

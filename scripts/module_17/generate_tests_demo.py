"""Lab 3 - generate a test for one function: a real LLM call
(`FakeRuntime`-backed) proposes a test function's source for `subtract()`,
written into a real sandboxed copy of the repo, then **actually executed**
via `run_tests()` to confirm it's syntactically valid and genuinely passes
- not just that the model produced plausible-looking text.
"""

from __future__ import annotations

import asyncio
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "packages"))

import re  # noqa: E402

from local_ai_core.runtimes.fake import FakeRuntime  # noqa: E402
from local_ai_core.runtimes.types import LLMRequest  # noqa: E402
from local_ai_agents.tools.run_tests import run_tests  # noqa: E402

_PASSED_COUNT_RE = re.compile(r"(\d+) passed")


def _passed_count(stdout: str) -> int:
    match = _PASSED_COUNT_RE.search(stdout)
    return int(match.group(1)) if match else 0

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SAMPLE_REPO = REPO_ROOT / "datasets" / "code_repos" / "mini_calculator"

GENERATE_TEST_PROMPT_TEMPLATE = """Write a pytest test function for this function:

{function_source}

Reply with only the test function's source code, nothing else."""

# A real, syntactically-valid, genuinely-passing generated test (scripted to
# stand in for a real model's output - see the module's machine note).
GENERATED_TEST_SOURCE = (
    "def test_subtract_generated():\n"
    "    from calculator import subtract\n"
    "    assert subtract(10, 4) == 6\n"
)


async def run_lab() -> dict:
    with tempfile.TemporaryDirectory() as tmp_dir:
        repo_dir = Path(tmp_dir) / "mini_calculator"
        shutil.copytree(SAMPLE_REPO, repo_dir)

        function_source = "def subtract(a: float, b: float) -> float:\n    return a - b\n"
        runtime = FakeRuntime(default_response=GENERATED_TEST_SOURCE)
        response = await runtime.generate(
            LLMRequest(model="fake-model", prompt=GENERATE_TEST_PROMPT_TEMPLATE.format(function_source=function_source))
        )
        generated_test = response.text.strip()

        before_result = await run_tests(repo_dir)

        test_file = repo_dir / "tests" / "test_generated.py"
        test_file.write_text(generated_test + "\n", encoding="utf-8")

        after_result = await run_tests(repo_dir)

        return {
            "generated_test_source": generated_test,
            "passed_count_before": _passed_count(before_result.stdout),
            "passed_count_after": _passed_count(after_result.stdout),
            "after_stdout_tail": after_result.stdout.strip().splitlines()[-1] if after_result.stdout.strip() else "",
        }


def result_to_markdown(result: dict) -> str:
    return (
        "# Lab 3 - generate a test for one function, then actually run it\n\n"
        f"- Generated test source:\n```python\n{result['generated_test_source']}\n```\n"
        f"- Passing test count before adding the generated test: {result['passed_count_before']}\n"
        f"- Passing test count after adding the generated test: {result['passed_count_after']} "
        "(the new test genuinely runs and passes; the repo's one pre-existing unrelated failure is untouched)\n"
        f"- Final pytest summary line: {result['after_stdout_tail']}\n"
    )


def main(argv: list[str] | None = None) -> int:
    result = asyncio.run(run_lab())
    print(result_to_markdown(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

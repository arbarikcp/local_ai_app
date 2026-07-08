"""Lab 1.3 — Small model failure analysis.

Gives a small (1B/3B-class) model five task types known to expose
small-model weaknesses (see docs/modules/01_local_llm_systems_thinking.md
section 11) and prints raw output for manual failure-mode annotation. This
lab is deliberately not auto-graded: the point is to *read* the failures,
not to reduce them to a pass/fail number this early in the course.

Usage:
    uv run python scripts/module_01/lab_1_3_small_model_failure_analysis.py --model qwen2.5:1.5b
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

from ollama_probe import OllamaUnavailable, generate, is_ollama_available

TASKS: dict[str, str] = {
    "strict_json_extraction": (
        "Extract the following fields as strict JSON only, with keys "
        '"name", "age", "city" and nothing else, no markdown fences, no '
        "commentary. If a field is missing, use null.\n\n"
        "Text: \"Maria moved to Austin last spring. She just turned 29.\""
    ),
    "multi_step_reasoning": (
        "A train leaves City A at 60 mph. Two hours later, a second train "
        "leaves City A on the same track at 90 mph, chasing the first. "
        "How many hours after the second train leaves does it catch the "
        "first? Show your reasoning, then give the final numeric answer on "
        "its own last line."
    ),
    "answer_with_citation": (
        "Context:\n"
        "[doc1] The Eiffel Tower was completed in 1889.\n"
        "[doc2] The Eiffel Tower is located in Paris, France.\n\n"
        "Question: When was the Eiffel Tower completed, and where is it "
        "located? Answer using only the context above, and cite the "
        "source doc id(s) you used in square brackets after each fact."
    ),
    "tool_argument_generation": (
        "You can call a function `get_weather(city: str, unit: 'celsius' | "
        "'fahrenheit')`. The user asks: \"What's the weather like in Tokyo "
        "right now, in Celsius?\" Respond with ONLY a JSON object matching "
        '{"function": "get_weather", "arguments": {...}}.'
    ),
    "code_patch_suggestion": (
        "This Python function has a bug: it should return the maximum "
        "value in a list, but returns the wrong thing for empty lists.\n\n"
        "def max_of(values):\n"
        "    m = 0\n"
        "    for v in values:\n"
        "        if v > m:\n"
        "            m = v\n"
        "    return m\n\n"
        "Give a corrected version of the function only, as a fenced python "
        "code block."
    ),
}


@dataclass(frozen=True)
class TaskResult:
    task_name: str
    prompt: str
    response_text: str


def run_lab(model: str, tasks: dict[str, str]) -> list[TaskResult]:
    results: list[TaskResult] = []
    for name, prompt in tasks.items():
        obs = generate(model, prompt)
        results.append(TaskResult(task_name=name, prompt=prompt, response_text=obs.response_text))
    return results


def results_to_markdown(model: str, results: list[TaskResult]) -> str:
    parts = [f"# Lab 1.3 — small model failure analysis\n\nModel: `{model}`\n"]
    for r in results:
        parts.append(f"## {r.task_name}\n")
        parts.append(f"**Prompt:**\n\n```text\n{r.prompt}\n```\n")
        parts.append(f"**Raw response:**\n\n```text\n{r.response_text}\n```\n")
        parts.append(
            "**Failure mode observed:** _(fill in manually — e.g. invalid "
            "JSON, markdown-wrapped output, dropped instruction, "
            "fabricated fact, wrong arithmetic, hallucinated tool name)_\n"
        )
    return "\n".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen2.5:1.5b")
    args = parser.parse_args(argv)

    if not is_ollama_available():
        print(
            "SKIPPED: Ollama is not reachable at http://localhost:11434. "
            "Install and start Ollama (Module 2) and re-run this lab.",
            file=sys.stderr,
        )
        return 1

    try:
        results = run_lab(args.model, TASKS)
    except OllamaUnavailable as exc:
        print(f"SKIPPED: {exc}", file=sys.stderr)
        return 1

    print(results_to_markdown(args.model, results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

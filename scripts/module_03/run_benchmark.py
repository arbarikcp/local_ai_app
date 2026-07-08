"""Module 3 benchmark harness — runs the 6 golden-set task suites against one
or more models and produces model scorecards.

Curriculum's illustrative deliverable path is ``model-eval-suite/runners/
run_benchmark.py``; this repo follows the ``scripts/module_NN/`` convention
established in Modules 1-2 instead (see
docs/modules/03_local_model_selection_and_benchmarking.md).

The generation call is injected as ``Callable[[str, str], str]`` (model,
prompt -> response text) so this whole harness is unit-testable without a
live model. Real usage plugs in ``default_generate_fn``, which calls Ollama
via Module 1's ``ollama_probe.py``.

Usage:
    uv run python scripts/module_03/run_benchmark.py --models qwen2.5:1.5b qwen2.5:3b qwen2.5:7b
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "module_01"))

from scorers.exact_match import contains_all, normalized_exact_match  # noqa: E402
from scorers.json_validity import has_required_keys, try_parse_json  # noqa: E402
from scorers.rag_metrics import answer_is_grounded_refusal, citation_validity  # noqa: E402

GenerateFn = Callable[[str, str], str]

GOLDEN_SETS_DIR = Path(__file__).resolve().parent.parent.parent / "evals" / "golden_sets"
TASK_FILES = [
    "summarization.jsonl",
    "extraction.jsonl",
    "classification.jsonl",
    "code.jsonl",
    "rag.jsonl",
    "tool_calling.jsonl",
]


def load_dataset(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def prompt_for_record(record: dict) -> str:
    task = record["task"]
    if task == "code":
        return record["prompt"]
    if task == "summarization":
        return f"Summarize the following text in 2-3 sentences:\n\n{record['text']}"
    if task == "extraction":
        keys = ", ".join(record["schema_keys"])
        return (
            f"Extract the following fields as strict JSON only ({keys}), using null for "
            f"missing fields. Do not include markdown.\n\nText: {record['text']}"
        )
    if task == "classification":
        labels = ", ".join(record["labels"])
        return (
            f"Classify the following text into exactly one of these labels: {labels}.\n"
            f"Respond with only the label.\n\nText: {record['text']}"
        )
    if task == "rag":
        context_block = "\n".join(f"[{d['doc_id']}] {d['text']}" for d in record["context"])
        return (
            "Answer only using the provided context, citing doc ids in square brackets. "
            'If the answer is not present, say "I don\'t know based on the provided documents."\n\n'
            f"Context:\n{context_block}\n\nQuestion: {record['question']}"
        )
    if task == "tool_calling":
        tools_desc = json.dumps(record["available_tools"])
        return (
            f"You can call these functions: {tools_desc}. The user says: "
            f'"{record["prompt"]}". If a function applies, respond with ONLY JSON '
            '{"function": ..., "arguments": {...}}. If none apply, respond with exactly: NO_TOOL'
        )
    raise ValueError(f"Don't know how to build a prompt for task: {task}")


def _loose_equal(a: object, b: object) -> bool:
    if isinstance(a, str) and isinstance(b, str):
        return a.strip().lower() == b.strip().lower()
    return a == b


def _score_tool_call(record: dict, prediction: str) -> float:
    parsed = try_parse_json(prediction)
    expected_tool = record.get("expected_tool")

    if expected_tool is None:
        # Negative case: the correct behavior is NOT producing a tool call.
        did_call_tool = isinstance(parsed, dict) and "function" in parsed
        return 0.0 if did_call_tool else 1.0

    if not isinstance(parsed, dict) or parsed.get("function") != expected_tool:
        return 0.0

    args = parsed.get("arguments", {})
    if not isinstance(args, dict):
        return 0.0

    expected_args = record.get("expected_arguments") or {}
    if not expected_args:
        return 1.0
    matches = sum(1 for k, v in expected_args.items() if k in args and _loose_equal(args[k], v))
    return matches / len(expected_args)


def score_record(record: dict, prediction: str) -> float:
    scorer = record["scorer"]
    if scorer == "contains_all":
        return 1.0 if contains_all(prediction, record["required_facts"]) else 0.0
    if scorer == "json_validity_and_keys":
        return 1.0 if has_required_keys(prediction, record["schema_keys"]) else 0.0
    if scorer == "normalized_exact_match":
        return 1.0 if normalized_exact_match(prediction, record["reference_label"]) else 0.0
    if scorer == "citation_validity":
        doc_ids = [d["doc_id"] for d in record["context"]]
        return citation_validity(prediction, doc_ids)
    if scorer == "grounded_refusal":
        return 1.0 if answer_is_grounded_refusal(prediction, record["refusal_phrase"]) else 0.0
    if scorer == "tool_call_validity":
        return _score_tool_call(record, prediction)
    raise ValueError(f"Unknown scorer: {scorer}")


@dataclass(frozen=True)
class RecordResult:
    record_id: str
    prediction: str
    score: float


@dataclass(frozen=True)
class TaskResult:
    task_name: str
    model: str
    record_results: list[RecordResult]

    @property
    def mean_score(self) -> float:
        if not self.record_results:
            return 0.0
        return sum(r.score for r in self.record_results) / len(self.record_results)


def run_task_benchmark(model: str, dataset: list[dict], generate_fn: GenerateFn) -> TaskResult:
    task_name = dataset[0]["task"] if dataset else "unknown"
    results = []
    for record in dataset:
        prompt = prompt_for_record(record)
        prediction = generate_fn(model, prompt)
        score = score_record(record, prediction)
        results.append(RecordResult(record_id=record["id"], prediction=prediction, score=score))
    return TaskResult(task_name=task_name, model=model, record_results=results)


def run_full_benchmark(
    models: list[str], dataset_paths: dict[str, Path], generate_fn: GenerateFn
) -> dict[str, list[TaskResult]]:
    results: dict[str, list[TaskResult]] = {}
    for model in models:
        model_results = []
        for _task_file, path in dataset_paths.items():
            dataset = load_dataset(path)
            model_results.append(run_task_benchmark(model, dataset, generate_fn))
        results[model] = model_results
    return results


def scorecard_quality_table(task_results: list[TaskResult]) -> str:
    header = "| Task | Score | Records |\n|---|---:|---:|\n"
    lines = [
        f"| {tr.task_name} | {tr.mean_score:.2f} | {len(tr.record_results)} |" for tr in task_results
    ]
    return header + "\n".join(lines)


def comparison_table(all_results: dict[str, list[TaskResult]]) -> str:
    """Model x task mean-score comparison table across all benchmarked models."""
    if not all_results:
        return ""
    task_names = [tr.task_name for tr in next(iter(all_results.values()))]
    header = "| Model | " + " | ".join(task_names) + " |\n"
    header += "|---|" + "|".join(["---:"] * len(task_names)) + "|\n"
    lines = []
    for model, task_results in all_results.items():
        scores = " | ".join(f"{tr.mean_score:.2f}" for tr in task_results)
        lines.append(f"| {model} | {scores} |")
    return header + "\n".join(lines)


def default_generate_fn(model: str, prompt: str) -> str:
    """Real generation via Ollama (Module 1's ollama_probe.py). Not called by
    unit tests — those inject a fake generate_fn instead.
    """
    from ollama_probe import generate as ollama_generate

    return ollama_generate(model, prompt).response_text


def main(argv: list[str] | None = None) -> int:
    from ollama_probe import is_ollama_available

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", required=True)
    args = parser.parse_args(argv)

    if not is_ollama_available():
        print(
            "SKIPPED: Ollama is not reachable at http://localhost:11434. "
            "This benchmark harness is built and unit-tested, but running it for real "
            "requires a resourced Mac with Ollama installed and the target models pulled. "
            "See docs/modules/03_local_model_selection_and_benchmarking.md.",
            file=sys.stderr,
        )
        return 1

    dataset_paths = {name: GOLDEN_SETS_DIR / name for name in TASK_FILES}
    results = run_full_benchmark(args.models, dataset_paths, default_generate_fn)

    print("# Benchmark comparison\n")
    print(comparison_table(results))
    for model, task_results in results.items():
        print(f"\n## {model}\n")
        print(scorecard_quality_table(task_results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

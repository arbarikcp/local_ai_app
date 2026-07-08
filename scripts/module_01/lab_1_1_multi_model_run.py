"""Lab 1.1 — Run the same prompt on multiple model sizes.

Usage:
    uv run python scripts/module_01/lab_1_1_multi_model_run.py \\
        --models qwen2.5:1.5b qwen2.5:3b qwen2.5:7b \\
        --prompt "Explain the difference between latency and throughput in two sentences."

Records model, quantization tag (as encoded in the Ollama tag name),
runtime, prompt/output tokens, TTFT, tokens/sec, and answer text for each
model, and writes a markdown table to stdout (redirect into a report file).

If Ollama is not reachable, this prints an explicit skip notice rather than
fabricating numbers — see docs/modules/01_local_llm_systems_thinking.md.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

from ollama_probe import GenerationObservation, OllamaUnavailable, generate, is_ollama_available

DEFAULT_MODELS = ["qwen2.5:1.5b", "qwen2.5:3b", "qwen2.5:7b"]
DEFAULT_PROMPT = (
    "Explain the difference between latency and throughput in two sentences."
)


@dataclass(frozen=True)
class LabRow:
    model: str
    prompt_tokens: int | None
    output_tokens: int | None
    ttft_seconds: float | None
    tokens_per_second: float | None
    wall_clock_seconds: float
    answer_preview: str


def run_lab(models: list[str], prompt: str) -> list[LabRow]:
    rows: list[LabRow] = []
    for model in models:
        obs: GenerationObservation = generate(model, prompt)
        preview = obs.response_text.strip().replace("\n", " ")[:160]
        rows.append(
            LabRow(
                model=model,
                prompt_tokens=obs.prompt_eval_count,
                output_tokens=obs.eval_count,
                ttft_seconds=obs.ttft_seconds,
                tokens_per_second=obs.tokens_per_second,
                wall_clock_seconds=obs.wall_clock_seconds,
                answer_preview=preview,
            )
        )
    return rows


def rows_to_markdown_table(rows: list[LabRow]) -> str:
    header = (
        "| Model | Prompt tokens | Output tokens | TTFT (s) | Tokens/sec | "
        "Wall clock (s) | Answer preview |\n"
        "|---|---:|---:|---:|---:|---:|---|\n"
    )
    lines = []
    for r in rows:
        lines.append(
            f"| {r.model} | {r.prompt_tokens} | {r.output_tokens} | "
            f"{_fmt(r.ttft_seconds)} | {_fmt(r.tokens_per_second)} | "
            f"{r.wall_clock_seconds:.2f} | {r.answer_preview} |"
        )
    return header + "\n".join(lines)


def _fmt(value: float | None) -> str:
    return f"{value:.2f}" if value is not None else "n/a"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    args = parser.parse_args(argv)

    if not is_ollama_available():
        print(
            "SKIPPED: Ollama is not reachable at http://localhost:11434. "
            "Install and start Ollama (Module 2), pull the requested models, "
            "and re-run this lab. No numbers were fabricated.",
            file=sys.stderr,
        )
        return 1

    try:
        rows = run_lab(args.models, args.prompt)
    except OllamaUnavailable as exc:
        print(f"SKIPPED: {exc}", file=sys.stderr)
        return 1

    print(f"# Lab 1.1 — multi-model comparison\n\nPrompt: `{args.prompt}`\n")
    print(rows_to_markdown_table(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

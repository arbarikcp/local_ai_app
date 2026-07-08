"""Lab 4.1 — Quantization comparison.

Runs the same task across different quantization tags of the same model
(e.g. Ollama tags like ``qwen2.5:7b-instruct-q4_K_M`` vs
``qwen2.5:7b-instruct-q8_0``) and captures quality, latency, memory, invalid
schema rate, and hallucination notes side by side.

Usage:
    uv run python scripts/module_04/lab_4_1_quantization_comparison.py \\
        --tags qwen2.5:7b-instruct-q4_K_M qwen2.5:7b-instruct-q8_0
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "module_01"))

from memory_sampler import find_pid_by_name, sample_peak_rss_during  # noqa: E402
from ollama_probe import OllamaUnavailable, generate, is_ollama_available  # noqa: E402

# A JSON-extraction task, deliberately reused from Module 1's failure-analysis
# lab: structured output is exactly where quantization-driven quality loss
# shows up first (theory doc §3).
DEFAULT_PROMPT = (
    'Extract the following fields as strict JSON only ("name", "age", "city"), '
    "no markdown, no commentary. If a field is missing, use null.\n\n"
    'Text: "Maria moved to Austin last spring. She just turned 29."'
)


@dataclass(frozen=True)
class QuantComparisonRow:
    tag: str
    response_text: str
    prompt_tokens: int | None
    output_tokens: int | None
    ttft_seconds: float | None
    tokens_per_second: float | None
    peak_rss_bytes: int | None


def run_lab(tags: list[str], prompt: str) -> list[QuantComparisonRow]:
    ollama_pid = find_pid_by_name("ollama")
    rows = []
    for tag in tags:
        if ollama_pid is not None:
            obs, peak = sample_peak_rss_during(ollama_pid, lambda t=tag: generate(t, prompt))
        else:
            obs, peak = generate(tag, prompt), None
        rows.append(
            QuantComparisonRow(
                tag=tag,
                response_text=obs.response_text,
                prompt_tokens=obs.prompt_eval_count,
                output_tokens=obs.eval_count,
                ttft_seconds=obs.ttft_seconds,
                tokens_per_second=obs.tokens_per_second,
                peak_rss_bytes=peak,
            )
        )
    return rows


def rows_to_markdown_table(rows: list[QuantComparisonRow]) -> str:
    header = (
        "| Tag | Prompt tokens | Output tokens | TTFT (s) | Tokens/sec | "
        "Peak RSS | Response |\n|---|---:|---:|---:|---:|---:|---|\n"
    )
    lines = []
    for r in rows:
        peak_str = f"{r.peak_rss_bytes / 1024**3:.2f} GiB" if r.peak_rss_bytes else "n/a"
        preview = r.response_text.strip().replace("\n", " ")[:100]
        lines.append(
            f"| {r.tag} | {r.prompt_tokens} | {r.output_tokens} | "
            f"{_fmt(r.ttft_seconds)} | {_fmt(r.tokens_per_second)} | {peak_str} | {preview} |"
        )
    return header + "\n".join(lines)


def _fmt(value: float | None) -> str:
    return f"{value:.2f}" if value is not None else "n/a"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tags", nargs="+", required=True, help="Ollama tags to compare")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    args = parser.parse_args(argv)

    if not is_ollama_available():
        print(
            "SKIPPED: Ollama is not reachable at http://localhost:11434. "
            "Pull the same model at multiple quantizations (e.g. `ollama pull "
            "qwen2.5:7b-instruct-q4_K_M` and `...q8_0`) and re-run on a resourced Mac.",
            file=sys.stderr,
        )
        return 1

    try:
        rows = run_lab(args.tags, args.prompt)
    except OllamaUnavailable as exc:
        print(f"SKIPPED: {exc}", file=sys.stderr)
        return 1

    print(f"# Lab 4.1 — quantization comparison\n\nPrompt: `{args.prompt}`\n")
    print(rows_to_markdown_table(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

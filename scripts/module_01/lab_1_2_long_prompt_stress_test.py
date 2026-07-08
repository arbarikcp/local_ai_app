"""Lab 1.2 — Long prompt stress test.

Runs the same model at increasing (approximate) prompt lengths — 500, 2000,
4000, 8000, 16000 tokens — and records latency, memory-relevant runtime
metrics, truncation, and error behavior.

This machine has no reliable way to sample macOS process RSS for a server
process it doesn't own without extra permissions, so peak memory is left as
a manual-observation column (Activity Monitor / `ps`) rather than faked.

Usage:
    uv run python scripts/module_01/lab_1_2_long_prompt_stress_test.py --model qwen2.5:3b
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

from ollama_probe import OllamaUnavailable, generate, is_ollama_available
from token_estimate import words_for_target_tokens

DEFAULT_TARGET_LENGTHS = [500, 2_000, 4_000, 8_000, 16_000]

# Filler content is repeated, semantically empty, and clearly labeled as
# such in the prompt so that any degradation observed is attributable to
# length/memory pressure rather than to a confusing task.
_FILLER_SENTENCE = (
    "The quick brown fox jumps over the lazy dog near the river bank. "
)


def build_prompt(target_tokens: int) -> str:
    words_needed = words_for_target_tokens(target_tokens)
    filler_words = _FILLER_SENTENCE.split()
    repeats = max(1, words_needed // len(filler_words) + 1)
    filler = (_FILLER_SENTENCE * repeats).split()
    filler_text = " ".join(filler[:words_needed])
    return (
        "Below is filler text of approximately "
        f"{target_tokens} tokens, followed by a question. Ignore the filler "
        "content; it is repeated nonsense used only to pad prompt length.\n\n"
        f"{filler_text}\n\n"
        "Question: What is the capital of France? Answer in one word."
    )


@dataclass(frozen=True)
class StressRow:
    target_tokens: int
    prompt_tokens: int | None
    output_tokens: int | None
    ttft_seconds: float | None
    tokens_per_second: float | None
    wall_clock_seconds: float
    truncated_or_errored: bool
    error: str | None
    answer_preview: str


def run_lab(model: str, target_lengths: list[int]) -> list[StressRow]:
    rows: list[StressRow] = []
    for target in target_lengths:
        prompt = build_prompt(target)
        try:
            obs = generate(model, prompt, timeout=600.0)
            rows.append(
                StressRow(
                    target_tokens=target,
                    prompt_tokens=obs.prompt_eval_count,
                    output_tokens=obs.eval_count,
                    ttft_seconds=obs.ttft_seconds,
                    tokens_per_second=obs.tokens_per_second,
                    wall_clock_seconds=obs.wall_clock_seconds,
                    truncated_or_errored=False,
                    error=None,
                    answer_preview=obs.response_text.strip().replace("\n", " ")[:120],
                )
            )
        except Exception as exc:  # noqa: BLE001 - we want to record ANY failure
            rows.append(
                StressRow(
                    target_tokens=target,
                    prompt_tokens=None,
                    output_tokens=None,
                    ttft_seconds=None,
                    tokens_per_second=None,
                    wall_clock_seconds=0.0,
                    truncated_or_errored=True,
                    error=str(exc),
                    answer_preview="",
                )
            )
    return rows


def rows_to_markdown_table(rows: list[StressRow]) -> str:
    header = (
        "| Target tokens | Actual prompt tokens | TTFT (s) | Tokens/sec | "
        "Wall clock (s) | Failed? | Notes |\n"
        "|---:|---:|---:|---:|---:|---|---|\n"
    )
    lines = []
    for r in rows:
        notes = r.error if r.error else r.answer_preview
        lines.append(
            f"| {r.target_tokens} | {r.prompt_tokens} | {_fmt(r.ttft_seconds)} | "
            f"{_fmt(r.tokens_per_second)} | {r.wall_clock_seconds:.2f} | "
            f"{'yes' if r.truncated_or_errored else 'no'} | {notes} |"
        )
    return header + "\n".join(lines)


def _fmt(value: float | None) -> str:
    return f"{value:.2f}" if value is not None else "n/a"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen2.5:3b")
    parser.add_argument("--target-lengths", nargs="+", type=int, default=DEFAULT_TARGET_LENGTHS)
    args = parser.parse_args(argv)

    if not is_ollama_available():
        print(
            "SKIPPED: Ollama is not reachable at http://localhost:11434. "
            "Install and start Ollama (Module 2) and re-run this lab.",
            file=sys.stderr,
        )
        return 1

    try:
        rows = run_lab(args.model, args.target_lengths)
    except OllamaUnavailable as exc:
        print(f"SKIPPED: {exc}", file=sys.stderr)
        return 1

    print(f"# Lab 1.2 — long prompt stress test\n\nModel: `{args.model}`\n")
    print(rows_to_markdown_table(rows))
    print(
        "\nPeak memory was not sampled automatically in this run — observe "
        "Activity Monitor or `ps -o rss -p <ollama_pid>` during a manual re-run "
        "and record it in the report."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

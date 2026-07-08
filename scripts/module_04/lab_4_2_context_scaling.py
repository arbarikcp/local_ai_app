"""Lab 4.2 — Context scaling, with memory sampled throughout.

Extends Module 1's Lab 1.2 (long-prompt stress test, which measured latency
only) by also sampling the runtime process's peak RSS at each context
length — turning "context scaling costs memory" (theory doc §5) into a
measured curve instead of just a formula.

Usage:
    uv run python scripts/module_04/lab_4_2_context_scaling.py --model qwen2.5:3b
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "module_01"))

from lab_1_2_long_prompt_stress_test import build_prompt  # noqa: E402
from memory_sampler import find_pid_by_name, sample_peak_rss_during  # noqa: E402
from ollama_probe import OllamaUnavailable, generate, is_ollama_available  # noqa: E402

DEFAULT_TARGET_LENGTHS = [2_000, 4_000, 8_000, 16_000]


@dataclass(frozen=True)
class ContextScalingRow:
    target_tokens: int
    actual_prompt_tokens: int | None
    wall_clock_seconds: float
    peak_rss_bytes: int | None
    error: str | None


def run_lab(model: str, target_lengths: list[int]) -> list[ContextScalingRow]:
    ollama_pid = find_pid_by_name("ollama")
    rows = []
    for target in target_lengths:
        prompt = build_prompt(target)
        try:
            if ollama_pid is not None:
                obs, peak = sample_peak_rss_during(
                    ollama_pid, lambda p=prompt: generate(model, p, timeout=600.0)
                )
            else:
                obs, peak = generate(model, prompt, timeout=600.0), None
            rows.append(
                ContextScalingRow(
                    target_tokens=target,
                    actual_prompt_tokens=obs.prompt_eval_count,
                    wall_clock_seconds=obs.wall_clock_seconds,
                    peak_rss_bytes=peak,
                    error=None,
                )
            )
        except Exception as exc:  # noqa: BLE001 - record any failure, don't crash the sweep
            rows.append(
                ContextScalingRow(
                    target_tokens=target,
                    actual_prompt_tokens=None,
                    wall_clock_seconds=0.0,
                    peak_rss_bytes=None,
                    error=str(exc),
                )
            )
    return rows


def rows_to_markdown_table(rows: list[ContextScalingRow]) -> str:
    header = (
        "| Target tokens | Actual prompt tokens | Wall clock (s) | Peak RSS | Error |\n"
        "|---:|---:|---:|---:|---|\n"
    )
    lines = []
    for r in rows:
        peak_str = f"{r.peak_rss_bytes / 1024**3:.2f} GiB" if r.peak_rss_bytes else "n/a"
        lines.append(
            f"| {r.target_tokens} | {r.actual_prompt_tokens} | {r.wall_clock_seconds:.2f} | "
            f"{peak_str} | {r.error or ''} |"
        )
    return header + "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen2.5:3b")
    parser.add_argument("--target-lengths", nargs="+", type=int, default=DEFAULT_TARGET_LENGTHS)
    args = parser.parse_args(argv)

    if not is_ollama_available():
        print(
            "SKIPPED: Ollama is not reachable at http://localhost:11434. "
            "Install Ollama (Module 2) and re-run this lab on a resourced Mac.",
            file=sys.stderr,
        )
        return 1

    try:
        rows = run_lab(args.model, args.target_lengths)
    except OllamaUnavailable as exc:
        print(f"SKIPPED: {exc}", file=sys.stderr)
        return 1

    print(f"# Lab 4.2 — context scaling\n\nModel: `{args.model}`\n")
    print(rows_to_markdown_table(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

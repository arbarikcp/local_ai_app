"""Lab 4.3 — Concurrency simulation.

Sends 1/2/4/8 concurrent requests to the same model and measures response
latency spread, timeout rate, and peak memory pressure for the whole batch —
the empirical half of theory doc §7-8 (batch size and concurrent requests
both multiply the KV-cache term directly).

Usage:
    uv run python scripts/module_04/lab_4_3_concurrency_simulation.py --model qwen2.5:3b
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "module_01"))

from memory_sampler import find_pid_by_name, sample_peak_rss_during  # noqa: E402
from ollama_probe import OllamaUnavailable, generate, is_ollama_available  # noqa: E402

DEFAULT_CONCURRENCY_LEVELS = [1, 2, 4, 8]
DEFAULT_PROMPT = "Explain what a KV cache is in two sentences."
DEFAULT_TIMEOUT_SECONDS = 60.0


@dataclass(frozen=True)
class ConcurrencyLevelResult:
    """Note: this lab cannot yet distinguish "timed out" from "failed for any
    other reason" — Module 1's ``ollama_probe.generate`` wraps every httpx
    error (connection refused, read timeout, ...) into the same
    ``OllamaUnavailable``, and a proper ``RequestTimeout`` vs. other-error
    taxonomy is Module 6's job (see curriculum.md §16's error taxonomy).
    ``failure_count``/``failure_rate`` here is honestly a superset of
    "timeout rate" until that lands, not a precise timeout measurement.
    """

    n_concurrent: int
    batch_wall_clock_seconds: float
    request_latencies_seconds: list[float] = field(default_factory=list)
    failure_count: int = 0
    peak_rss_bytes: int | None = None

    @property
    def mean_latency_seconds(self) -> float | None:
        if not self.request_latencies_seconds:
            return None
        return sum(self.request_latencies_seconds) / len(self.request_latencies_seconds)

    @property
    def max_latency_seconds(self) -> float | None:
        return max(self.request_latencies_seconds) if self.request_latencies_seconds else None

    @property
    def failure_rate(self) -> float:
        total = self.n_concurrent
        return self.failure_count / total if total else 0.0


def _fire_one_request(model: str, prompt: str, timeout: float) -> float:
    """Returns wall-clock latency in seconds for a single request."""
    start = time.perf_counter()
    generate(model, prompt, timeout=timeout)
    return time.perf_counter() - start


def run_concurrency_level(
    model: str, prompt: str, n_concurrent: int, timeout: float = DEFAULT_TIMEOUT_SECONDS
) -> ConcurrencyLevelResult:
    ollama_pid = find_pid_by_name("ollama")
    latencies: list[float] = []
    failure_count = 0

    def run_batch() -> None:
        nonlocal failure_count
        with ThreadPoolExecutor(max_workers=n_concurrent) as pool:
            futures = [pool.submit(_fire_one_request, model, prompt, timeout) for _ in range(n_concurrent)]
            for future in as_completed(futures):
                try:
                    latencies.append(future.result())
                except Exception:  # noqa: BLE001 - count and continue, don't abort the batch
                    failure_count += 1

    batch_start = time.perf_counter()
    if ollama_pid is not None:
        _, peak = sample_peak_rss_during(ollama_pid, run_batch)
    else:
        run_batch()
        peak = None
    batch_wall_clock = time.perf_counter() - batch_start

    return ConcurrencyLevelResult(
        n_concurrent=n_concurrent,
        batch_wall_clock_seconds=batch_wall_clock,
        request_latencies_seconds=latencies,
        failure_count=failure_count,
        peak_rss_bytes=peak,
    )


def run_lab(
    model: str, prompt: str, levels: list[int], timeout: float = DEFAULT_TIMEOUT_SECONDS
) -> list[ConcurrencyLevelResult]:
    return [run_concurrency_level(model, prompt, n, timeout) for n in levels]


def results_to_markdown_table(results: list[ConcurrencyLevelResult]) -> str:
    header = (
        "| Concurrent requests | Batch wall clock (s) | Mean latency (s) | Max latency (s) | "
        "Failure rate | Peak RSS |\n|---:|---:|---:|---:|---:|---:|\n"
    )
    lines = []
    for r in results:
        peak_str = f"{r.peak_rss_bytes / 1024**3:.2f} GiB" if r.peak_rss_bytes else "n/a"
        lines.append(
            f"| {r.n_concurrent} | {r.batch_wall_clock_seconds:.2f} | "
            f"{_fmt(r.mean_latency_seconds)} | {_fmt(r.max_latency_seconds)} | "
            f"{r.failure_rate:.0%} | {peak_str} |"
        )
    return header + "\n".join(lines)


def _fmt(value: float | None) -> str:
    return f"{value:.2f}" if value is not None else "n/a"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen2.5:3b")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--levels", nargs="+", type=int, default=DEFAULT_CONCURRENCY_LEVELS)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    args = parser.parse_args(argv)

    if not is_ollama_available():
        print(
            "SKIPPED: Ollama is not reachable at http://localhost:11434. "
            "Install Ollama (Module 2) and re-run this lab on a resourced Mac. "
            "Watch for thermal throttling symptoms (fan noise, slowdown over the run) "
            "manually - this script does not detect that automatically.",
            file=sys.stderr,
        )
        return 1

    try:
        results = run_lab(args.model, args.prompt, args.levels, args.timeout)
    except OllamaUnavailable as exc:
        print(f"SKIPPED: {exc}", file=sys.stderr)
        return 1

    print(f"# Lab 4.3 — concurrency simulation\n\nModel: `{args.model}`\n")
    print(results_to_markdown_table(results))
    print(
        "\nNote: thermal throttling symptoms (fan noise, sustained slowdown) must be "
        "observed manually during the run and noted in the report."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

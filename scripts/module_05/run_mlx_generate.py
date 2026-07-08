"""Lab 1/5/6 deliverable — repeatable MLX generation with warmup and
streaming, distinct from Module 2's basic MLX smoke test (which only asked
"does it work at all"). This asks the Module 5 questions: is the first call
slower than later ones, and can it stream?

Reuses Module 2's Apple-Silicon/importability checks rather than
duplicating them (docs/modules/02_mac_local_ai_development_environment.md).
The summary-building logic is pure and unit-tested; the actual model
load/generate calls are real network/compute calls, honest-skip on this
machine per this repo's established pattern.

Usage:
    uv run python scripts/module_05/run_mlx_generate.py \\
        --model mlx-community/Qwen2.5-1.5B-Instruct-4bit --prompt "Say hello in five words."
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "module_02"))

from smoke_test_mlx import (  # noqa: E402
    INSTALL_INSTRUCTIONS,
    NOT_APPLE_SILICON_MESSAGE,
    check_mlx_importable,
    is_apple_silicon,
)

DEFAULT_MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
DEFAULT_PROMPT = "Say hello in five words."
DEFAULT_MAX_TOKENS = 64


@dataclass(frozen=True)
class MlxRunSummary:
    model_id: str
    load_seconds: float
    cold_generate_seconds: float
    warm_generate_seconds: float
    stream_ttft_seconds: float | None
    stream_total_seconds: float | None
    cold_text: str
    warm_text: str
    streamed_text: str

    @property
    def warmup_speedup_factor(self) -> float | None:
        if self.warm_generate_seconds == 0:
            return None
        return self.cold_generate_seconds / self.warm_generate_seconds


def build_summary(
    *,
    model_id: str,
    load_seconds: float,
    cold_seconds: float,
    warm_seconds: float,
    stream_ttft_seconds: float | None,
    stream_total_seconds: float | None,
    cold_text: str,
    warm_text: str,
    streamed_text: str,
) -> MlxRunSummary:
    return MlxRunSummary(
        model_id=model_id,
        load_seconds=load_seconds,
        cold_generate_seconds=cold_seconds,
        warm_generate_seconds=warm_seconds,
        stream_ttft_seconds=stream_ttft_seconds,
        stream_total_seconds=stream_total_seconds,
        cold_text=cold_text,
        warm_text=warm_text,
        streamed_text=streamed_text,
    )


def summary_to_markdown(summary: MlxRunSummary) -> str:
    speedup = summary.warmup_speedup_factor
    speedup_str = f"{speedup:.2f}x" if speedup is not None else "n/a"
    ttft_str = f"{summary.stream_ttft_seconds:.3f}s" if summary.stream_ttft_seconds is not None else "n/a"
    stream_total_str = (
        f"{summary.stream_total_seconds:.2f}s" if summary.stream_total_seconds is not None else "n/a"
    )
    return (
        f"# MLX run — {summary.model_id}\n\n"
        f"- Load time: {summary.load_seconds:.2f}s\n"
        f"- Cold generate: {summary.cold_generate_seconds:.2f}s\n"
        f"- Warm generate: {summary.warm_generate_seconds:.2f}s\n"
        f"- Warmup speedup (cold/warm): {speedup_str}\n"
        f"- Streaming TTFT: {ttft_str}\n"
        f"- Streaming total: {stream_total_str}\n"
    )


def run_warmup_and_streaming_demo(
    model_id: str, prompt: str, max_tokens: int = DEFAULT_MAX_TOKENS
) -> MlxRunSummary:
    """Real MLX model load + generate calls. Not unit tested directly (no MLX
    model available on this machine) - build_summary()/summary_to_markdown()
    carry the tested logic; this function is the thin real-compute wrapper.
    """
    from mlx_lm import generate, load, stream_generate

    load_start = time.perf_counter()
    mlx_model, tokenizer = load(model_id)
    load_seconds = time.perf_counter() - load_start

    cold_start = time.perf_counter()
    cold_text = generate(mlx_model, tokenizer, prompt=prompt, max_tokens=max_tokens)
    cold_seconds = time.perf_counter() - cold_start

    warm_start = time.perf_counter()
    warm_text = generate(mlx_model, tokenizer, prompt=prompt, max_tokens=max_tokens)
    warm_seconds = time.perf_counter() - warm_start

    stream_start = time.perf_counter()
    first_token_time: float | None = None
    streamed_parts: list[str] = []
    for response in stream_generate(mlx_model, tokenizer, prompt=prompt, max_tokens=max_tokens):
        if first_token_time is None:
            first_token_time = time.perf_counter()
        streamed_parts.append(response.text)
    stream_total_seconds = time.perf_counter() - stream_start
    stream_ttft = (first_token_time - stream_start) if first_token_time is not None else None

    return build_summary(
        model_id=model_id,
        load_seconds=load_seconds,
        cold_seconds=cold_seconds,
        warm_seconds=warm_seconds,
        stream_ttft_seconds=stream_ttft,
        stream_total_seconds=stream_total_seconds,
        cold_text=cold_text,
        warm_text=warm_text,
        streamed_text="".join(streamed_parts),
    )


def main(argv: list[str] | None = None) -> int:
    import argparse
    import platform

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    args = parser.parse_args(argv)

    if not is_apple_silicon():
        print(NOT_APPLE_SILICON_MESSAGE.format(arch=platform.machine()), file=sys.stderr)
        return 1

    if not check_mlx_importable():
        print(INSTALL_INSTRUCTIONS, file=sys.stderr)
        return 1

    summary = run_warmup_and_streaming_demo(args.model, args.prompt, args.max_tokens)
    print(summary_to_markdown(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

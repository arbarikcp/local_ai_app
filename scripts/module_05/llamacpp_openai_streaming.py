"""Lab 3/4 — streaming through an OpenAI-compatible local server.

Extends Module 2's non-streaming llama-cpp-python server smoke test to
streaming, using the standard `openai` Python client's `stream=True` chat
completion — the same client code path an application would use against a
hosted API, pointed at a fully local server instead (theory doc §4).

The accumulator (``accumulate_chat_stream``) is pure and unit-tested against
plain fixture tuples rather than real OpenAI SDK stream objects, so it does
not require a live server or the `openai` package to test the logic that
actually matters. ``stream_chat_completion`` is the thin real-network
wrapper, honest-skip on this machine.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable

DEFAULT_BASE_URL = "http://localhost:8080/v1"
DEFAULT_PROMPT = "Count from 1 to 5, one number per line."


@dataclass(frozen=True)
class AccumulatedChatStream:
    full_text: str
    chunk_count: int
    ttft_seconds: float | None
    total_seconds: float | None


def accumulate_chat_stream(
    text_chunks_with_timestamps: Iterable[tuple[str, float]], start_time: float
) -> AccumulatedChatStream:
    parts: list[str] = []
    count = 0
    ttft: float | None = None
    last_timestamp = start_time
    for text, timestamp in text_chunks_with_timestamps:
        count += 1
        if ttft is None and text:
            ttft = timestamp - start_time
        parts.append(text)
        last_timestamp = timestamp
    total = (last_timestamp - start_time) if count else None
    return AccumulatedChatStream(
        full_text="".join(parts), chunk_count=count, ttft_seconds=ttft, total_seconds=total
    )


def check_openai_client_importable() -> bool:
    try:
        import openai  # noqa: F401

        return True
    except ImportError:
        return False


def stream_chat_completion(
    model: str, prompt: str, *, base_url: str = DEFAULT_BASE_URL, timeout: float = 300.0
) -> AccumulatedChatStream:
    """Real streaming call against a live OpenAI-compatible server."""
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key="not-needed-for-local-server", timeout=timeout)
    start = time.perf_counter()
    stream = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}], stream=True
    )
    chunks_with_timestamps: list[tuple[str, float]] = []
    for chunk in stream:
        delta_text = ""
        if chunk.choices and chunk.choices[0].delta.content:
            delta_text = chunk.choices[0].delta.content
        chunks_with_timestamps.append((delta_text, time.perf_counter()))
    return accumulate_chat_stream(chunks_with_timestamps, start)


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--model", default="local-gguf-model")
    args = parser.parse_args(argv)

    if not check_openai_client_importable():
        print("SKIPPED: the `openai` package is not installed. Run: uv add openai", file=sys.stderr)
        return 1

    from openai import APIConnectionError

    try:
        result = stream_chat_completion(args.model, args.prompt, base_url=args.base_url)
    except APIConnectionError:
        print(
            f"SKIPPED: could not reach an OpenAI-compatible server at {args.base_url}. "
            "Start one with scripts/module_05/serve_llamacpp.sh on a resourced Mac.",
            file=sys.stderr,
        )
        return 1

    print(f"# Streaming chat completion — {args.base_url}\n")
    print(f"Prompt: `{args.prompt}`")
    print(f"Response: {result.full_text}")
    print(f"Chunks: {result.chunk_count}")
    print(f"TTFT: {result.ttft_seconds:.3f}s" if result.ttft_seconds is not None else "TTFT: n/a")
    print(f"Total: {result.total_seconds:.3f}s" if result.total_seconds is not None else "Total: n/a")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

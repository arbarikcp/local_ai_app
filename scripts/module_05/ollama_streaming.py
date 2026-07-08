"""Lab 2/4/5 — Ollama native streaming, real TTFT, and cancellation.

Ollama's /api/generate streams newline-delimited JSON (NDJSON): one JSON
object per line, with an incremental "response" text fragment, until a
final object carrying "done": true and the same usage fields as the
non-streaming endpoint.

The parsing logic below (``parse_stream_line``, ``accumulate_stream``) is
pure and fully unit-tested against fixture NDJSON lines shaped like real
Ollama output. The network-calling functions (``stream_generate``,
``generate_with_cancellation``) are thin wrappers around that logic and are
honest-skip on a machine with no reachable Ollama server, per this repo's
established pattern (docs/modules/01_local_llm_systems_thinking.md).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator

import httpx

DEFAULT_BASE_URL = "http://localhost:11434"


@dataclass(frozen=True)
class StreamChunk:
    """One parsed line of an Ollama streaming response."""

    text: str
    done: bool
    eval_count: int | None = None
    eval_duration_ns: int | None = None
    raw: dict[str, Any] = field(default_factory=dict)


def parse_stream_line(raw_line: str) -> StreamChunk | None:
    """Parse a single NDJSON line from Ollama's streaming response.

    Returns None for blank lines (which the transport may emit between
    chunks) rather than raising, since a blank line is not an error.
    """
    stripped = raw_line.strip()
    if not stripped:
        return None
    data = json.loads(stripped)
    return StreamChunk(
        text=data.get("response", ""),
        done=bool(data.get("done", False)),
        eval_count=data.get("eval_count"),
        eval_duration_ns=data.get("eval_duration"),
        raw=data,
    )


def iter_stream_chunks(raw_lines: Iterable[str]) -> Iterator[StreamChunk]:
    """Parse a sequence of raw NDJSON lines into StreamChunks, skipping blanks."""
    for raw_line in raw_lines:
        chunk = parse_stream_line(raw_line)
        if chunk is not None:
            yield chunk


@dataclass(frozen=True)
class AccumulatedStream:
    full_text: str
    chunk_count: int
    ttft_seconds: float | None
    total_seconds: float | None
    eval_count: int | None
    tokens_per_second: float | None


def accumulate_stream(
    chunks_with_timestamps: Iterable[tuple[StreamChunk, float]], start_time: float
) -> AccumulatedStream:
    """Fold a sequence of (chunk, arrival_time) pairs into a summary.

    ``start_time`` is the wall-clock time the request was sent (the
    ``time.perf_counter()`` value right before the request), so that
    ``ttft_seconds`` measures real time-to-first-token rather than an
    approximation (unlike Module 1's non-streaming TTFT).
    """
    text_parts: list[str] = []
    chunk_count = 0
    ttft_seconds: float | None = None
    last_timestamp = start_time
    final_eval_count: int | None = None
    final_eval_duration_ns: int | None = None

    for chunk, arrival_time in chunks_with_timestamps:
        chunk_count += 1
        if ttft_seconds is None and chunk.text:
            ttft_seconds = arrival_time - start_time
        text_parts.append(chunk.text)
        last_timestamp = arrival_time
        if chunk.done:
            final_eval_count = chunk.eval_count
            final_eval_duration_ns = chunk.eval_duration_ns

    total_seconds = (last_timestamp - start_time) if chunk_count else None
    tokens_per_second = None
    if final_eval_count and final_eval_duration_ns:
        tokens_per_second = final_eval_count / (final_eval_duration_ns / 1e9)

    return AccumulatedStream(
        full_text="".join(text_parts),
        chunk_count=chunk_count,
        ttft_seconds=ttft_seconds,
        total_seconds=total_seconds,
        eval_count=final_eval_count,
        tokens_per_second=tokens_per_second,
    )


def stream_generate(
    model: str, prompt: str, *, base_url: str = DEFAULT_BASE_URL, timeout: float = 300.0
) -> AccumulatedStream:
    """Real streaming call against a live Ollama server. Honest-skip via
    httpx raising if unreachable - callers should check is_ollama_available()
    first (see ollama_probe.py) for a clean skip message instead of a raw
    exception reaching the user.
    """
    payload = {"model": model, "prompt": prompt, "stream": True}
    start = time.perf_counter()
    chunks_with_timestamps: list[tuple[StreamChunk, float]] = []
    with httpx.stream("POST", f"{base_url}/api/generate", json=payload, timeout=timeout) as response:
        response.raise_for_status()
        for raw_line in response.iter_lines():
            chunk = parse_stream_line(raw_line)
            if chunk is not None:
                chunks_with_timestamps.append((chunk, time.perf_counter()))
    return accumulate_stream(chunks_with_timestamps, start)


@dataclass(frozen=True)
class CancellationResult:
    tokens_received_before_cancel: int
    elapsed_seconds_before_cancel: float
    text_received: str


def generate_with_cancellation(
    model: str,
    prompt: str,
    cancel_after_tokens: int,
    *,
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = 300.0,
) -> CancellationResult:
    """Start a streaming request and close the connection early, after
    ``cancel_after_tokens`` chunks arrive.

    Caveat (see theory doc's Gotchas): this measures client-side elapsed
    time at the point of cancellation. It is a proxy for "the server
    stopped working," not direct proof the server freed the compute -
    verifying that would need server-side instrumentation this course does
    not have access to.
    """
    payload = {"model": model, "prompt": prompt, "stream": True}
    start = time.perf_counter()
    received_count = 0
    text_parts: list[str] = []
    with httpx.stream("POST", f"{base_url}/api/generate", json=payload, timeout=timeout) as response:
        response.raise_for_status()
        for raw_line in response.iter_lines():
            chunk = parse_stream_line(raw_line)
            if chunk is None:
                continue
            received_count += 1
            text_parts.append(chunk.text)
            if received_count >= cancel_after_tokens:
                break  # closing the `with` block below closes the connection early
    elapsed = time.perf_counter() - start
    return CancellationResult(
        tokens_received_before_cancel=received_count,
        elapsed_seconds_before_cancel=elapsed,
        text_received="".join(text_parts),
    )

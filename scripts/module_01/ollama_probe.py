"""Minimal Ollama HTTP probe used only by Module 1 labs.

This is intentionally NOT the reusable ``LLMRuntime`` abstraction — that is
defined once, canonically, in Module 6
(``packages/local_ai_core/runtimes/``). Module 1 is a pre-abstraction module:
its job is to observe raw runtime behavior, not to design the interface.
Keeping this file lab-local avoids pulling Module 1 into Module 6's package
boundary.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import httpx

DEFAULT_BASE_URL = "http://localhost:11434"


class OllamaUnavailable(RuntimeError):
    """Raised when the local Ollama server cannot be reached."""


@dataclass(frozen=True)
class GenerationObservation:
    """Raw, unprocessed measurements from a single local generation call.

    Field names mirror what Ollama's ``/api/generate`` response actually
    returns so lab reports record measured values, not estimates.
    """

    model: str
    prompt: str
    response_text: str
    prompt_eval_count: int | None
    eval_count: int | None
    total_duration_ns: int | None
    load_duration_ns: int | None
    prompt_eval_duration_ns: int | None
    eval_duration_ns: int | None
    wall_clock_seconds: float
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def ttft_seconds(self) -> float | None:
        """Time to first token, approximated as load + prompt-eval duration.

        Ollama's non-streaming ``/api/generate`` response does not expose a
        true first-token timestamp, so this is the best available proxy from
        that endpoint. Streaming mode (not used here) would give an exact
        TTFT and should be preferred when precision matters.
        """
        if self.load_duration_ns is None or self.prompt_eval_duration_ns is None:
            return None
        return (self.load_duration_ns + self.prompt_eval_duration_ns) / 1e9

    @property
    def tokens_per_second(self) -> float | None:
        if not self.eval_count or self.eval_duration_ns is None or self.eval_duration_ns == 0:
            return None
        return self.eval_count / (self.eval_duration_ns / 1e9)


def is_ollama_available(base_url: str = DEFAULT_BASE_URL, timeout: float = 1.5) -> bool:
    try:
        resp = httpx.get(f"{base_url}/api/tags", timeout=timeout)
        return resp.status_code == 200
    except httpx.HTTPError:
        return False


def list_local_models(base_url: str = DEFAULT_BASE_URL, timeout: float = 5.0) -> list[str]:
    resp = httpx.get(f"{base_url}/api/tags", timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return [m["name"] for m in data.get("models", [])]


def generate(
    model: str,
    prompt: str,
    *,
    base_url: str = DEFAULT_BASE_URL,
    temperature: float = 0.0,
    timeout: float = 300.0,
) -> GenerationObservation:
    """Call Ollama's non-streaming generate endpoint and record raw metrics."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }
    start = time.perf_counter()
    try:
        resp = httpx.post(f"{base_url}/api/generate", json=payload, timeout=timeout)
    except httpx.HTTPError as exc:
        raise OllamaUnavailable(f"Could not reach Ollama at {base_url}: {exc}") from exc
    wall_clock = time.perf_counter() - start
    resp.raise_for_status()
    data = resp.json()

    return GenerationObservation(
        model=model,
        prompt=prompt,
        response_text=data.get("response", ""),
        prompt_eval_count=data.get("prompt_eval_count"),
        eval_count=data.get("eval_count"),
        total_duration_ns=data.get("total_duration"),
        load_duration_ns=data.get("load_duration"),
        prompt_eval_duration_ns=data.get("prompt_eval_duration"),
        eval_duration_ns=data.get("eval_duration"),
        wall_clock_seconds=wall_clock,
        raw=data,
    )

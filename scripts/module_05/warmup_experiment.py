"""Lab 5 — warmup experiment: cold vs. warm TTFT.

Theory doc §6: the first request to a freshly-loaded model is measurably
slower than subsequent requests (allocator warm-up, compute-graph
construction, Metal shader compilation on first use). This measures that
directly rather than asserting it from folklore.

The TTFT-measuring call is injected as ``Callable[[str, str], float | None]``
so the orchestration/statistics logic here is fully testable without a live
model - real usage passes a function backed by ``ollama_probe.generate`` or
``ollama_streaming.stream_generate`` (the latter gives a true, not
approximated, TTFT - see theory doc §3).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

TtftFn = Callable[[str, str], "float | None"]


@dataclass(frozen=True)
class WarmupResult:
    model: str
    cold_ttft_seconds: float | None
    warm_ttft_seconds: list[float | None] = field(default_factory=list)

    @property
    def successful_warm_ttfts(self) -> list[float]:
        return [t for t in self.warm_ttft_seconds if t is not None]

    @property
    def mean_warm_ttft_seconds(self) -> float | None:
        successes = self.successful_warm_ttfts
        if not successes:
            return None
        return sum(successes) / len(successes)

    @property
    def speedup_factor(self) -> float | None:
        """cold_ttft / mean_warm_ttft. >1 means warm requests were faster."""
        mean_warm = self.mean_warm_ttft_seconds
        if self.cold_ttft_seconds is None or mean_warm is None or mean_warm == 0:
            return None
        return self.cold_ttft_seconds / mean_warm


def run_warmup_experiment(
    model: str, prompt: str, get_ttft_fn: TtftFn, n_warm_calls: int = 3
) -> WarmupResult:
    if n_warm_calls < 1:
        raise ValueError("n_warm_calls must be >= 1")
    cold = get_ttft_fn(model, prompt)
    warm = [get_ttft_fn(model, prompt) for _ in range(n_warm_calls)]
    return WarmupResult(model=model, cold_ttft_seconds=cold, warm_ttft_seconds=warm)


def result_to_markdown(result: WarmupResult) -> str:
    warm_str = ", ".join(f"{t:.3f}s" if t is not None else "failed" for t in result.warm_ttft_seconds)
    cold_str = f"{result.cold_ttft_seconds:.3f}s" if result.cold_ttft_seconds is not None else "failed"
    mean_warm = result.mean_warm_ttft_seconds
    mean_warm_str = f"{mean_warm:.3f}s" if mean_warm is not None else "n/a"
    speedup = result.speedup_factor
    speedup_str = f"{speedup:.2f}x" if speedup is not None else "n/a"
    return (
        f"# Warmup experiment — {result.model}\n\n"
        f"- Cold TTFT: {cold_str}\n"
        f"- Warm TTFTs: {warm_str}\n"
        f"- Mean warm TTFT: {mean_warm_str}\n"
        f"- Speedup (cold / mean warm): {speedup_str}\n"
    )

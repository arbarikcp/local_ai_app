"""Lab 4.4 — Predict, then measure.

The module's core deliverable lab. For a chosen model (looked up in
model_shapes.py), computes predicted peak memory at 2K/8K/16K context using
memory_math.py, then actually measures peak memory via memory_sampler.py
while running a real generation call at each context length. Produces the
prediction-vs-actual table the module's assessment requires.

Usage:
    uv run python scripts/module_04/lab_4_4_predict_then_measure.py \\
        --model-tag qwen2.5:7b-instruct-q4_K_M --shape qwen2.5-7b --quant Q4_K_M
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "module_01"))

from lab_1_2_long_prompt_stress_test import build_prompt  # noqa: E402
from memory_math import estimate_memory_budget  # noqa: E402
from memory_sampler import find_pid_by_name, sample_peak_rss_during  # noqa: E402
from model_shapes import get_shape  # noqa: E402
from ollama_probe import OllamaUnavailable, generate, is_ollama_available  # noqa: E402

DEFAULT_CONTEXT_LENGTHS = [2_000, 8_000, 16_000]

GAP_EXPLANATION_CHECKLIST = (
    "fill in manually - discuss: allocator/runtime overhead beyond the planning-grade "
    "0.5-1.5GB range, unified-memory accounting (shared with OS/other apps, not isolated "
    "VRAM), background apps resident at measurement time, and runtime-specific buffering "
    "(e.g. Ollama's own process overhead vs raw llama.cpp)"
)


@dataclass(frozen=True)
class PredictVsActualRow:
    model_tag: str
    quant: str
    context_tokens: int
    predicted_low_gib: float
    predicted_high_gib: float
    actual_peak_gib: float | None
    gap_explanation: str


def predict_and_measure(
    model_tag: str,
    shape_id: str,
    quant: str,
    context_lengths: list[int],
    kv_quant: str = "FP16",
) -> list[PredictVsActualRow]:
    shape = get_shape(shape_id)
    ollama_pid = find_pid_by_name("ollama")
    rows = []
    for context_tokens in context_lengths:
        estimate = estimate_memory_budget(
            n_params=shape.n_params,
            quant=quant,
            n_layers=shape.n_layers,
            n_kv_heads=shape.n_kv_heads,
            head_dim=shape.head_dim,
            context_tokens=context_tokens,
            kv_quant=kv_quant,
        )
        prompt = build_prompt(context_tokens)

        actual_peak_gib: float | None = None
        if ollama_pid is not None:
            _, peak_bytes = sample_peak_rss_during(
                ollama_pid, lambda p=prompt: generate(model_tag, p, timeout=600.0)
            )
            if peak_bytes is not None:
                actual_peak_gib = peak_bytes / 1024**3
        else:
            generate(model_tag, prompt, timeout=600.0)

        rows.append(
            PredictVsActualRow(
                model_tag=model_tag,
                quant=quant,
                context_tokens=context_tokens,
                predicted_low_gib=estimate.total_low_gib,
                predicted_high_gib=estimate.total_high_gib,
                actual_peak_gib=actual_peak_gib,
                gap_explanation=GAP_EXPLANATION_CHECKLIST,
            )
        )
    return rows


def rows_to_markdown_table(rows: list[PredictVsActualRow]) -> str:
    header = (
        "| Model | Quant | Context | Predicted memory | Actual peak memory | Gap explanation |\n"
        "|---|---|---:|---:|---:|---|\n"
    )
    lines = []
    for r in rows:
        predicted = f"{r.predicted_low_gib:.1f}-{r.predicted_high_gib:.1f} GB/GiB"
        actual = f"{r.actual_peak_gib:.2f} GiB" if r.actual_peak_gib is not None else "not measured"
        lines.append(
            f"| {r.model_tag} | {r.quant} | {r.context_tokens} | {predicted} | {actual} | "
            f"{r.gap_explanation} |"
        )
    return header + "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-tag", required=True, help="Runtime tag, e.g. qwen2.5:7b-instruct-q4_K_M")
    parser.add_argument("--shape", required=True, help="Key into model_shapes.KNOWN_SHAPES, e.g. qwen2.5-7b")
    parser.add_argument("--quant", default="Q4_K_M")
    parser.add_argument("--kv-quant", default="FP16")
    parser.add_argument("--context-lengths", nargs="+", type=int, default=DEFAULT_CONTEXT_LENGTHS)
    args = parser.parse_args(argv)

    if not is_ollama_available():
        print(
            "SKIPPED: Ollama is not reachable at http://localhost:11434. "
            "This is the module's core lab - predicted values can be computed offline "
            "(see the notebook), but actual measurement needs a resourced Mac.",
            file=sys.stderr,
        )
        return 1

    try:
        rows = predict_and_measure(args.model_tag, args.shape, args.quant, args.context_lengths, args.kv_quant)
    except OllamaUnavailable as exc:
        print(f"SKIPPED: {exc}", file=sys.stderr)
        return 1

    print(f"# Lab 4.4 — predict, then measure\n\nModel: `{args.model_tag}` (shape: {args.shape})\n")
    print(rows_to_markdown_table(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Lab 8 — compare prompt-only+retry vs. JSON-schema-constrained vs.
grammar-constrained extraction on invalid-JSON rate, field accuracy, and
p95 latency.

Uses the golden extraction set from Module 3 (evals/golden_sets/
extraction.jsonl), filtered to the 2 records that match PersonExtraction's
schema (name/age/city) - the other 4 records use different schemas
(invoice, contact, medical, order) not modeled by this module's example
schema. 2 real records is a thin sample; documented honestly in the report
rather than padded with records that don't semantically fit.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "packages"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "module_01"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "module_06_5"))

from lab_measure_concurrency import percentile  # noqa: E402
from local_ai_core.extraction.pipeline import ExtractionPipeline, RequestedFormat  # noqa: E402
from local_ai_core.extraction.schemas import PersonExtraction  # noqa: E402

GOLDEN_SET_PATH = Path(__file__).resolve().parent.parent.parent / "evals" / "golden_sets" / "extraction.jsonl"
PERSON_SCHEMA_KEYS = {"name", "age", "city"}
MODES: list[RequestedFormat] = ["text", "json_schema", "grammar"]


def load_matching_golden_cases(schema_keys: set[str] = PERSON_SCHEMA_KEYS, path: Path = GOLDEN_SET_PATH) -> list[dict]:
    """Golden-set records whose schema_keys exactly match the given schema."""
    cases = []
    with open(path) as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            if set(record["schema_keys"]) == schema_keys:
                cases.append(record)
    return cases


def field_accuracy(predicted: dict, reference: dict) -> float:
    """Fraction of reference fields the prediction got exactly right,
    including correctly-predicted nulls (Module 3's ext-005 case: not
    fabricating a missing age is a correct prediction, not a miss).
    """
    if not reference:
        return 0.0
    correct = sum(1 for key, value in reference.items() if predicted.get(key) == value)
    return correct / len(reference)


@dataclass(frozen=True)
class ModeResult:
    mode: str
    invalid_json_rate: float
    field_accuracy: float
    p95_latency_seconds: float
    used_constrained_decoding_rate: float


async def run_mode(mode: RequestedFormat, runtime, cases: list[dict], model: str) -> ModeResult:
    pipeline = ExtractionPipeline(runtime, PersonExtraction, response_format_type=mode, max_repair_attempts=1)
    invalid_count = 0
    accuracies = []
    latencies = []
    constrained_count = 0

    for case in cases:
        start = time.perf_counter()
        result = await pipeline.run(case["text"], model)
        latencies.append(time.perf_counter() - start)
        if result.used_constrained_decoding:
            constrained_count += 1
        if result.parsed is None:
            invalid_count += 1
        else:
            accuracies.append(field_accuracy(result.fields, case["reference"]))

    return ModeResult(
        mode=mode,
        invalid_json_rate=invalid_count / len(cases) if cases else 0.0,
        field_accuracy=sum(accuracies) / len(accuracies) if accuracies else 0.0,
        p95_latency_seconds=percentile(latencies, 0.95),
        used_constrained_decoding_rate=constrained_count / len(cases) if cases else 0.0,
    )


async def run_lab(runtime, model: str, cases: list[dict] | None = None) -> list[ModeResult]:
    cases = cases if cases is not None else load_matching_golden_cases()
    return [await run_mode(mode, runtime, cases, model) for mode in MODES]


def results_to_markdown_table(results: list[ModeResult]) -> str:
    header = (
        "| Mode | Invalid JSON rate | Field accuracy | p95 latency (s) | Constrained decoding used |\n"
        "|---|---:|---:|---:|---:|\n"
    )
    lines = [
        f"| {r.mode} | {r.invalid_json_rate:.0%} | {r.field_accuracy:.0%} | "
        f"{r.p95_latency_seconds:.3f} | {r.used_constrained_decoding_rate:.0%} |"
        for r in results
    ]
    return header + "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    from ollama_probe import is_ollama_available
    from local_ai_core.runtimes.ollama import OllamaRuntime

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen2.5:1.5b")
    args = parser.parse_args(argv)

    if not is_ollama_available():
        print(
            "SKIPPED: Ollama is not reachable at http://localhost:11434. "
            "Install Ollama (Module 2) and re-run this lab on the resourced 32GB Mac. "
            "Note: real Ollama has no grammar support (Module 5's feature_matrix.py), so "
            "the 'grammar' mode row will show 0% constrained-decoding-used even when this "
            "lab runs for real - that is expected, not a bug.",
            file=sys.stderr,
        )
        return 1

    cases = load_matching_golden_cases()

    async def _run():
        runtime = OllamaRuntime()
        try:
            return await run_lab(runtime, args.model, cases)
        finally:
            await runtime.aclose()

    results = asyncio.run(_run())
    print(f"# Lab 8 — constrained decoding comparison\n\nModel: `{args.model}`, {len(cases)} golden cases\n")
    print(results_to_markdown_table(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

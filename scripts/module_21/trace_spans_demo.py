"""Lab 3 - add real trace spans. Drives a full (fake) request through
curriculum's exact trace-model steps, then proves the resulting trace has
every core-required step with real, positive elapsed time.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "packages"))

from local_ai_core.tracing.trace import TraceBuilder, validate_trace_shape  # noqa: E402


def run_lab() -> dict:
    builder = TraceBuilder(request_id="req-demo-001")

    with builder.span("input_validation", schema="TicketClassificationRequest"):
        time.sleep(0.001)

    with builder.span("prompt_template_version", version="v3"):
        pass

    builder.record_retrieval_step(
        query="how do I reset my password?",
        chunk_ids=["password-reset-guide::0"],
        reranker_scores=[0.92],
    )

    with builder.span("context_packing", packed_chunk_count=1):
        pass

    with builder.span("model_call", model="ticket-classifier"):
        time.sleep(0.002)

    with builder.span("output_validation", valid=True):
        pass

    with builder.span("final_response", text="account"):
        pass

    with builder.span("evaluation_hooks", must_contain_score=1.0):
        pass

    trace = builder.build()
    missing = validate_trace_shape(trace)

    return {
        "request_id": trace.request_id,
        "span_names": trace.span_names(),
        "total_elapsed_ms": trace.total_elapsed_ms(),
        "missing_core_steps": missing,
    }


def result_to_markdown(result: dict) -> str:
    return (
        "# Lab 3 - trace spans\n\n"
        f"- Request ID: {result['request_id']}\n"
        f"- Span order: {' -> '.join(result['span_names'])}\n"
        f"- Total elapsed: {result['total_elapsed_ms']:.2f}ms\n"
        f"- Missing core steps: {result['missing_core_steps']}\n"
    )


def main(argv: list[str] | None = None) -> int:
    result = run_lab()
    print(result_to_markdown(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

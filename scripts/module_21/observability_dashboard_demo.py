"""Lab 6 - build a local dashboard/report. Drives a handful of (fake)
requests end to end - traces, metrics, eval scores, and user feedback all
tied together by `trace_id` - then prints one combined observability
report, the same role Module 20's `PerformanceDashboard` played for
latency alone, extended here to the full request lifecycle.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "packages"))

from local_ai_core.tracing.eval_feedback_store import (  # noqa: E402
    EvalFeedbackStore,
    EvalRunRecord,
    UserFeedbackRecord,
)
from local_ai_core.tracing.metrics_registry import MetricsRegistry  # noqa: E402
from local_ai_core.tracing.trace import TraceBuilder, validate_trace_shape  # noqa: E402

REQUESTS = [
    {"trace_id": "req-001", "eval_score": 1.0, "feedback": "up"},
    {"trace_id": "req-002", "eval_score": 0.5, "feedback": "down"},
    {"trace_id": "req-003", "eval_score": 1.0, "feedback": "up"},
]


def run_lab() -> dict:
    registry = MetricsRegistry()
    store = EvalFeedbackStore()
    traces = []

    for spec in REQUESTS:
        builder = TraceBuilder(request_id=spec["trace_id"])
        with builder.span("input_validation"):
            pass
        with builder.span("model_call", model="ticket-classifier"):
            pass
        with builder.span("final_response"):
            pass
        trace = builder.build()
        traces.append(trace)

        registry.increment("request_count")
        registry.observe("request_latency_ms", trace.total_elapsed_ms())

        store.log_eval_run(EvalRunRecord(trace_id=spec["trace_id"], metric_name="must_contain_score", score=spec["eval_score"]))
        store.log_user_feedback(UserFeedbackRecord(trace_id=spec["trace_id"], rating=spec["feedback"]))

    metrics_summary = registry.summary()
    feedback_summary = store.feedback_summary()
    incomplete_traces = [t.request_id for t in traces if validate_trace_shape(t)]
    store.close()

    return {
        "request_count": metrics_summary["request_count"].count,
        "mean_latency_ms": metrics_summary["request_latency_ms"].mean,
        "feedback_summary": feedback_summary,
        "incomplete_traces": incomplete_traces,
    }


def result_to_markdown(result: dict) -> str:
    return (
        "# Lab 6 - observability dashboard\n\n"
        f"- Requests traced: {result['request_count']}\n"
        f"- Mean trace latency: {result['mean_latency_ms']:.4f}ms\n"
        f"- Feedback summary: {result['feedback_summary']}\n"
        f"- Traces missing a core step: {result['incomplete_traces']}\n"
    )


def main(argv: list[str] | None = None) -> int:
    result = run_lab()
    print(result_to_markdown(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

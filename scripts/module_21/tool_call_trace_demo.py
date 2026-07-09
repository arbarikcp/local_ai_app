"""Lab 5 - trace tool calls. Records a successful and a failing tool call
into one trace via `record_tool_call_step()`, and increments the real
`tool_call_count`/`tool_error_count` metrics alongside it - the trace and
the metrics agree on how many tool calls happened and how many failed.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "packages"))

from local_ai_core.tracing.metrics_registry import MetricsRegistry  # noqa: E402
from local_ai_core.tracing.trace import TraceBuilder  # noqa: E402


def run_lab() -> dict:
    registry = MetricsRegistry()
    builder = TraceBuilder(request_id="req-tool-001")

    builder.record_tool_call_step(
        tool_name="lookup_order", arguments={"order_id": "ORD-123"}, result={"status": "shipped"}
    )
    registry.increment("tool_call_count")

    builder.record_tool_call_step(
        tool_name="cancel_order", arguments={"order_id": "ORD-999"}, error="order not found"
    )
    registry.increment("tool_call_count")
    registry.increment("tool_error_count")

    trace = builder.build()
    summary = registry.summary()

    return {
        "tool_call_spans": [s.attributes for s in trace.spans if s.name == "tool_calls"],
        "tool_call_count": summary["tool_call_count"].count,
        "tool_error_count": summary["tool_error_count"].count,
    }


def result_to_markdown(result: dict) -> str:
    lines = ["# Lab 5 - trace tool calls", ""]
    for span in result["tool_call_spans"]:
        status = "error" if span["error"] else "ok"
        lines.append(f"- {span['tool_name']}({span['arguments']}) -> {status}")
    lines.append(
        f"\nMetrics: tool_call_count={result['tool_call_count']}, tool_error_count={result['tool_error_count']}"
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    result = run_lab()
    print(result_to_markdown(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

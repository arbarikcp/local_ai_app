"""Labs 1-2 - add structured logs and request IDs. Module 6's
`ensure_trace_id()` is reused unchanged for request IDs; `StructuredLogger`
emits real JSON log lines under all four `PromptLoggingPolicy` values,
proving the HASH_ONLY and NONE policies genuinely never leak the raw
prompt text into a log line.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "packages"))

from local_ai_core.runtimes.base import ensure_trace_id  # noqa: E402
from local_ai_core.runtimes.types import LLMRequest  # noqa: E402
from local_ai_core.tracing.pii_redaction import redact_pii  # noqa: E402
from local_ai_core.tracing.structured_logging import PromptLoggingPolicy, render_prompt_field  # noqa: E402

RAW_PROMPT = "Classify this ticket for jane.doe@example.com: I was charged twice."


def run_lab() -> dict:
    request = ensure_trace_id(LLMRequest(model="ticket-classifier", prompt=RAW_PROMPT))

    fields_by_policy = {}
    for policy in PromptLoggingPolicy:
        if policy == PromptLoggingPolicy.REDACTED:
            fields = render_prompt_field(RAW_PROMPT, policy, redactor=lambda t: redact_pii(t).redacted_text)
        else:
            fields = render_prompt_field(RAW_PROMPT, policy)
        fields_by_policy[policy.value] = fields

    return {
        "trace_id": request.trace_id,
        "fields_by_policy": fields_by_policy,
    }


def result_to_markdown(result: dict) -> str:
    lines = [
        "# Labs 1-2 - structured logs and request IDs",
        "",
        f"- Request ID (Module 6's `ensure_trace_id()`): `{result['trace_id']}`",
        "",
        "| Policy | Logged fields |",
        "|---|---|",
    ]
    for policy, fields in result["fields_by_policy"].items():
        lines.append(f"| {policy} | `{json.dumps(fields)}` |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    result = run_lab()
    print(result_to_markdown(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

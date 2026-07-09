"""StructuredLogger and PromptLoggingPolicy (theory doc §1, §4) — real
structured JSON log emission via stdlib `logging` (same "a metrics hook
that logs *is* structured logging" precedent Module 6's `LoggingMetricsHook`
established), plus an explicit, named decision for how much of a prompt a
log line may contain - never a bare default, the same discipline Module
6.5's `AdmissionPolicy` applied to concurrency.
"""

from __future__ import annotations

import hashlib
import json
import logging
from enum import Enum
from typing import Any, Callable

Redactor = Callable[[str], str]


class PromptLoggingPolicy(Enum):
    FULL = "full"
    REDACTED = "redacted"
    HASH_ONLY = "hash_only"
    NONE = "none"


def _hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def render_prompt_field(
    prompt: str, policy: PromptLoggingPolicy, *, redactor: Redactor | None = None
) -> dict[str, Any]:
    """Turns a raw prompt into exactly what a log line is allowed to
    contain under `policy` - the decision is made once, here, so no caller
    can accidentally log a full prompt under a REDACTED/NONE policy by
    forgetting to apply it themselves.
    """
    if policy == PromptLoggingPolicy.FULL:
        return {"prompt": prompt, "prompt_logging_policy": policy.value}
    if policy == PromptLoggingPolicy.REDACTED:
        if redactor is None:
            raise ValueError("REDACTED policy requires a redactor function")
        return {"prompt": redactor(prompt), "prompt_logging_policy": policy.value}
    if policy == PromptLoggingPolicy.HASH_ONLY:
        return {"prompt_hash": _hash_prompt(prompt), "prompt_logging_policy": policy.value}
    return {"prompt_logging_policy": policy.value}


class StructuredLogger:
    """Emits one JSON object per event via stdlib `logging`, with a stable
    field set (`trace_id`, `event`, plus arbitrary `fields`) - a log
    consumer can always parse the message as JSON, never has to
    string-match a hand-formatted sentence.
    """

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger or logging.getLogger("local_ai_core.tracing")

    def log_event(self, event: str, *, trace_id: str, fields: dict[str, Any] | None = None) -> None:
        record = {"event": event, "trace_id": trace_id, **(fields or {})}
        self.logger.info(json.dumps(record, sort_keys=True))

    def log_prompt(
        self,
        prompt: str,
        *,
        trace_id: str,
        policy: PromptLoggingPolicy,
        redactor: Redactor | None = None,
    ) -> None:
        fields = render_prompt_field(prompt, policy, redactor=redactor)
        self.log_event("prompt_logged", trace_id=trace_id, fields=fields)

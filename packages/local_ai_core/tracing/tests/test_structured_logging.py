import json
import logging

import pytest

from local_ai_core.tracing.structured_logging import (
    PromptLoggingPolicy,
    StructuredLogger,
    render_prompt_field,
)


class TestRenderPromptField:
    def test_full_policy_includes_the_raw_prompt(self):
        fields = render_prompt_field("my secret prompt", PromptLoggingPolicy.FULL)
        assert fields["prompt"] == "my secret prompt"

    def test_redacted_policy_applies_the_redactor(self):
        fields = render_prompt_field(
            "call me at 555-1234", PromptLoggingPolicy.REDACTED, redactor=lambda t: t.replace("555-1234", "[PHONE]")
        )
        assert fields["prompt"] == "call me at [PHONE]"

    def test_redacted_policy_without_a_redactor_raises(self):
        with pytest.raises(ValueError):
            render_prompt_field("text", PromptLoggingPolicy.REDACTED)

    def test_hash_only_policy_never_includes_the_raw_prompt(self):
        fields = render_prompt_field("my secret prompt", PromptLoggingPolicy.HASH_ONLY)
        assert "prompt" not in fields
        assert "my secret prompt" not in json.dumps(fields)
        assert len(fields["prompt_hash"]) == 16

    def test_hash_only_policy_is_deterministic(self):
        first = render_prompt_field("same text", PromptLoggingPolicy.HASH_ONLY)
        second = render_prompt_field("same text", PromptLoggingPolicy.HASH_ONLY)
        assert first["prompt_hash"] == second["prompt_hash"]

    def test_none_policy_omits_the_prompt_entirely(self):
        fields = render_prompt_field("my secret prompt", PromptLoggingPolicy.NONE)
        assert "prompt" not in fields
        assert "prompt_hash" not in fields


class TestStructuredLogger:
    def test_log_event_emits_valid_json_with_trace_id_and_event(self, caplog):
        logger = StructuredLogger()
        with caplog.at_level(logging.INFO, logger="local_ai_core.tracing"):
            logger.log_event("request_started", trace_id="trace-123", fields={"model": "test-model"})

        record = json.loads(caplog.records[0].message)
        assert record["event"] == "request_started"
        assert record["trace_id"] == "trace-123"
        assert record["model"] == "test-model"

    def test_log_prompt_under_hash_only_never_leaks_the_raw_prompt(self, caplog):
        logger = StructuredLogger()
        with caplog.at_level(logging.INFO, logger="local_ai_core.tracing"):
            logger.log_prompt(
                "my secret prompt", trace_id="trace-456", policy=PromptLoggingPolicy.HASH_ONLY
            )

        assert "my secret prompt" not in caplog.records[0].message

    def test_accepts_a_custom_logger(self):
        custom_logger = logging.getLogger("my.custom.tracing.logger")
        logger = StructuredLogger(logger=custom_logger)
        assert logger.logger is custom_logger

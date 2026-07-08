import pytest
from pydantic import ValidationError

from local_ai_core.runtimes.types import LLMRequest, LLMResponse, ResponseFormat


class TestResponseFormat:
    def test_defaults_to_text(self):
        fmt = ResponseFormat()
        assert fmt.type == "text"
        assert fmt.schema_ is None
        assert fmt.grammar is None

    def test_accepts_schema_via_alias(self):
        fmt = ResponseFormat(type="json_schema", schema={"type": "object"})
        assert fmt.schema_ == {"type": "object"}

    def test_accepts_schema_via_field_name_too(self):
        # populate_by_name=True means both the alias and the field name work.
        fmt = ResponseFormat(type="json_schema", schema_={"type": "object"})
        assert fmt.schema_ == {"type": "object"}

    def test_rejects_unknown_type(self):
        with pytest.raises(ValidationError):
            ResponseFormat(type="not-a-real-type")


class TestLLMRequest:
    def test_minimal_request_has_sane_defaults(self):
        req = LLMRequest(model="qwen2.5:1.5b", prompt="hi")
        assert req.system is None
        assert req.temperature == 0.0
        assert req.max_tokens == 512
        assert req.stop == []
        assert req.response_format.type == "text"
        assert req.trace_id is None
        assert req.metadata == {}

    def test_requires_model_and_prompt(self):
        with pytest.raises(ValidationError):
            LLMRequest(prompt="hi")  # missing model
        with pytest.raises(ValidationError):
            LLMRequest(model="m")  # missing prompt

    def test_full_request_round_trips(self):
        req = LLMRequest(
            model="qwen2.5:7b",
            system="You are terse.",
            prompt="Summarize this.",
            temperature=0.2,
            max_tokens=100,
            stop=["\n\n"],
            response_format=ResponseFormat(type="json_schema", schema={"type": "object"}),
            trace_id="abc-123",
            metadata={"user_id": "u1"},
        )
        assert req.system == "You are terse."
        assert req.temperature == 0.2
        assert req.stop == ["\n\n"]
        assert req.response_format.type == "json_schema"
        assert req.trace_id == "abc-123"
        assert req.metadata == {"user_id": "u1"}

    def test_is_serializable_to_dict_with_schema_alias(self):
        req = LLMRequest(
            model="m", prompt="p", response_format=ResponseFormat(type="json_schema", schema={"a": 1})
        )
        dumped = req.model_dump(by_alias=True)
        assert dumped["response_format"]["schema"] == {"a": 1}


class TestLLMResponse:
    def test_minimal_response_has_sane_defaults(self):
        resp = LLMResponse(text="hello", model="m")
        assert resp.prompt_tokens is None
        assert resp.completion_tokens is None
        assert resp.latency_ms is None
        assert resp.stop_reason is None
        assert resp.raw == {}

    def test_full_response_round_trips(self):
        resp = LLMResponse(
            text="hello",
            model="m",
            prompt_tokens=10,
            completion_tokens=5,
            latency_ms=123.4,
            stop_reason="stop",
            raw={"eval_count": 5},
        )
        assert resp.prompt_tokens == 10
        assert resp.completion_tokens == 5
        assert resp.latency_ms == pytest.approx(123.4)
        assert resp.raw == {"eval_count": 5}

    def test_requires_text_and_model(self):
        with pytest.raises(ValidationError):
            LLMResponse(model="m")
        with pytest.raises(ValidationError):
            LLMResponse(text="hi")

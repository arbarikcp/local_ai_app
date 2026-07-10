import pytest
from pydantic import ValidationError

from gw_schemas import (
    BenchmarkRequest,
    BenchmarkResultResponse,
    GenerateRequest,
    GenerateResponse,
    StreamRequest,
)


class TestGenerateRequest:
    def test_requires_task_and_prompt(self):
        with pytest.raises(ValidationError):
            GenerateRequest.model_validate({"task": "chat"})

    def test_accepts_a_valid_request(self):
        request = GenerateRequest.model_validate({"task": "chat", "prompt": "hello"})
        assert request.task == "chat"


class TestGenerateResponse:
    def test_shape(self):
        response = GenerateResponse.model_validate(
            {"answer": "hi", "model_used": "qwen2.5:1.5b-instruct", "used_fallback": True, "trace_id": "t-1"}
        )
        assert response.used_fallback is True


class TestStreamRequest:
    def test_requires_task_and_prompt(self):
        with pytest.raises(ValidationError):
            StreamRequest.model_validate({"prompt": "hello"})


class TestBenchmarkRequest:
    def test_defaults_task_to_none_and_repeats_to_three(self):
        request = BenchmarkRequest.model_validate({})
        assert request.task is None
        assert request.repeats == 3


class TestBenchmarkResultResponse:
    def test_shape(self):
        result = BenchmarkResultResponse.model_validate(
            {
                "name": "chat-primary",
                "sample_count": 3,
                "mean_latency_ms": 10.0,
                "p95_latency_ms": 12.0,
                "mean_tokens_per_second": 20.0,
            }
        )
        assert result.sample_count == 3

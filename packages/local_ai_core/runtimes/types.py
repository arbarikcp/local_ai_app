"""Request/response types for the canonical LLMRuntime abstraction.

This is THE definition of these types for the whole course (curriculum.md
§16). Matches the curriculum's spec exactly - do not redefine these in a
later module; extend by adding new adapters, not new request/response
shapes.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

ResponseFormatType = Literal["text", "json_schema", "grammar"]


class ResponseFormat(BaseModel):
    type: ResponseFormatType = "text"
    schema_: dict[str, Any] | None = Field(default=None, alias="schema")
    grammar: str | None = None

    model_config = {"populate_by_name": True}


class LLMRequest(BaseModel):
    model: str
    system: str | None = None
    prompt: str
    temperature: float = 0.0
    max_tokens: int = 512
    stop: list[str] = Field(default_factory=list)
    response_format: ResponseFormat = Field(default_factory=ResponseFormat)
    trace_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class LLMResponse(BaseModel):
    text: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_ms: float | None = None
    stop_reason: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict)

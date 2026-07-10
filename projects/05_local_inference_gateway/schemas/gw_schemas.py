"""API request/response shapes for the gateway (ARCHITECTURE.md "API
contract").
"""

from __future__ import annotations

from pydantic import BaseModel


class GenerateRequest(BaseModel):
    task: str
    prompt: str


class GenerateResponse(BaseModel):
    answer: str
    model_used: str
    used_fallback: bool
    trace_id: str


class StreamRequest(BaseModel):
    task: str
    prompt: str


class BenchmarkRequest(BaseModel):
    task: str | None = None
    repeats: int = 3


class BenchmarkResultResponse(BaseModel):
    name: str
    sample_count: int
    mean_latency_ms: float
    p95_latency_ms: float
    mean_tokens_per_second: float

"""The FastAPI service (ARCHITECTURE.md "API contract"). Same lazy-context
pattern as Module 23's `api_app.py` and Projects 1-4's own `*_api.py`
files (`get_ctx()` builds and caches on `app.state.ctx` on first use, never
at import time).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT.parent.parent / "packages"))
sys.path.insert(0, str(_PROJECT_ROOT / "app"))
sys.path.insert(0, str(_PROJECT_ROOT / "schemas"))
sys.path.insert(0, str(_PROJECT_ROOT / "prompts"))
sys.path.insert(0, str(_PROJECT_ROOT / "evals"))

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import StreamingResponse  # noqa: E402

from local_ai_core.deployment.config import AppConfig, load_config  # noqa: E402
from local_ai_core.deployment.health import run_liveness_check, run_readiness_check  # noqa: E402
from local_ai_core.gateway.queue import QueueFullError  # noqa: E402
from local_ai_core.optimization.fallback import NoRuntimesAvailable  # noqa: E402
from local_ai_core.runtimes.errors import RequestTimeout  # noqa: E402

from gw_router import TaskNotFoundError  # noqa: E402
from gw_schemas import (  # noqa: E402
    BenchmarkRequest,
    BenchmarkResultResponse,
    GenerateRequest,
    GenerateResponse,
    StreamRequest,
)
from gw_service import GatewayAppContext, build_gw_context, run_generate, run_gw_benchmark, run_stream  # noqa: E402

REPO_ROOT = _PROJECT_ROOT.parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "app.example.yaml"
DEFAULT_CATALOG_PATH = REPO_ROOT / "models" / "MODEL_CATALOG.md"
DEFAULT_ROUTES_PATH = _PROJECT_ROOT / "config" / "gateway_routes.yaml"


def _build_default_context() -> GatewayAppContext:
    config_path = os.environ.get("APP_CONFIG_PATH", str(DEFAULT_CONFIG_PATH))
    config: AppConfig = load_config(config_path)
    return build_gw_context(config, model_catalog_path=DEFAULT_CATALOG_PATH, routes_path=DEFAULT_ROUTES_PATH)


app = FastAPI(title="Local Inference Gateway (Project 5)")


def get_ctx() -> GatewayAppContext:
    if getattr(app.state, "ctx", None) is None:
        app.state.ctx = _build_default_context()
    return app.state.ctx


@app.get("/health")
def health() -> dict:
    result = run_liveness_check()
    return {"status": "ok" if result.passed else "fail", "detail": result.detail}


@app.get("/ready")
def ready() -> dict:
    ctx = get_ctx()
    result = run_readiness_check(ctx.base.data_dir, ctx.base.model_registry)
    if not result.passed:
        raise HTTPException(status_code=503, detail=result.detail)
    return {"status": "ready", "detail": result.detail}


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest) -> GenerateResponse:
    ctx = get_ctx()
    try:
        result = await run_generate(ctx, request.task, request.prompt)
    except TaskNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except QueueFullError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except RequestTimeout as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except NoRuntimesAvailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return GenerateResponse(
        answer=result.answer, model_used=result.model_used, used_fallback=result.used_fallback, trace_id=result.trace_id
    )


@app.post("/stream")
async def stream(request: StreamRequest) -> StreamingResponse:
    ctx = get_ctx()
    try:
        chunk_iterator = run_stream(ctx, request.task, request.prompt)
    except TaskNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return StreamingResponse(chunk_iterator, media_type="text/plain")


@app.get("/requests/{request_id}")
def get_request(request_id: str) -> dict:
    ctx = get_ctx()
    record = ctx.storage.get_request(request_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"no request found with request_id {request_id!r}")
    return {
        "request_id": record.request_id,
        "trace_id": record.trace_id,
        "task": record.task,
        "model_used": record.model_used,
        "used_fallback": record.used_fallback,
        "latency_ms": record.latency_ms,
        "status": record.status,
        "created_at": record.created_at,
    }


@app.post("/benchmark", response_model=list[BenchmarkResultResponse])
async def benchmark(request: BenchmarkRequest) -> list[BenchmarkResultResponse]:
    ctx = get_ctx()
    try:
        results = await run_gw_benchmark(ctx, task=request.task, repeats=request.repeats)
    except TaskNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return [
        BenchmarkResultResponse(
            name=r.name, sample_count=r.sample_count, mean_latency_ms=r.mean_latency_ms,
            p95_latency_ms=r.p95_latency_ms, mean_tokens_per_second=r.mean_tokens_per_second,
        )
        for r in results
    ]

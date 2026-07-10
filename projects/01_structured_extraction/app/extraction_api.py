"""The FastAPI service (ARCHITECTURE.md "API contract"). Copies Module
23's `api_app.py` lazy-context pattern exactly (`get_ctx()` builds and
caches on `app.state.ctx` on first use, never at import time, so tests can
inject a test-scoped context first) and adds this project's four
endpoints on top.
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

from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from local_ai_core.deployment.config import AppConfig, load_config  # noqa: E402
from local_ai_core.deployment.health import run_liveness_check, run_readiness_check  # noqa: E402
from local_ai_core.gateway.queue import QueueFullError  # noqa: E402
from local_ai_core.runtimes.errors import RequestTimeout, RuntimeUnavailable  # noqa: E402

from extraction_normalization import TextTooLongError  # noqa: E402
from extraction_prompts import SchemaNotFoundError  # noqa: E402
from extraction_service import ExtractionAppContext, build_extraction_context, run_extraction  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "app.example.yaml"
DEFAULT_CATALOG_PATH = REPO_ROOT / "models" / "MODEL_CATALOG.md"

# ~4 characters per token is a common rough heuristic (distinct from this
# course's word-based ~1.3 tokens/word heuristic used elsewhere) - good
# enough for a char-length request cap, never trusted for a real token
# budget decision.
_CHARS_PER_TOKEN_HEURISTIC = 4


def _build_default_context() -> ExtractionAppContext:
    config_path = os.environ.get("APP_CONFIG_PATH", str(DEFAULT_CONFIG_PATH))
    config: AppConfig = load_config(config_path)
    return build_extraction_context(config, model_catalog_path=DEFAULT_CATALOG_PATH)


app = FastAPI(title="Local Structured Extraction Service (Project 1)")


def get_ctx() -> ExtractionAppContext:
    if getattr(app.state, "ctx", None) is None:
        app.state.ctx = _build_default_context()
    return app.state.ctx


class ExtractRequest(BaseModel):
    schema_name: str
    text: str


class ExtractResponse(BaseModel):
    request_id: str
    status: str
    data: dict
    confidence: str
    validation_errors: list[str]
    latency_ms: float


def _record_to_response(record) -> ExtractResponse:
    status = "needs_review" if record.needs_review else "success"
    return ExtractResponse(
        request_id=record.request_id,
        status=status,
        data=record.extracted_fields,
        confidence=record.confidence,
        validation_errors=[record.validation_error] if record.validation_error else [],
        latency_ms=record.latency_ms,
    )


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


@app.post("/extract", response_model=ExtractResponse)
async def extract(request: ExtractRequest) -> ExtractResponse:
    ctx = get_ctx()
    max_input_chars = ctx.base.config.limits.max_prompt_tokens * _CHARS_PER_TOKEN_HEURISTIC

    async def _run():
        return await run_extraction(
            ctx,
            schema_name=request.schema_name,
            text=request.text,
            max_input_chars=max_input_chars,
        )

    try:
        queued_result = await ctx.base.admission_controller.submit(_run)
    except QueueFullError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except SchemaNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except TextTooLongError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except RequestTimeout as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except RuntimeUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return _record_to_response(queued_result.result)


@app.get("/extractions/low-confidence")
def low_confidence(limit: int = 50) -> list[dict]:
    ctx = get_ctx()
    records = ctx.storage.list_low_confidence(limit=limit)
    return [_record_to_response(r).model_dump() for r in records]


@app.get("/extractions/{request_id}")
def get_extraction(request_id: str) -> dict:
    ctx = get_ctx()
    record = ctx.storage.get(request_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"no extraction found with request_id {request_id!r}")
    return _record_to_response(record).model_dump()

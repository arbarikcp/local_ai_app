"""Lab 2 - package local API. A real `FastAPI()` app: `/health` (liveness),
`/ready` (readiness), `/models` (registry), `/chat` (guarded by Module 22's
classifier, admitted through Module 6.5's `AdmissionController`, answered
by the injected runtime). Tested via `fastapi.testclient.TestClient`, which
drives the real ASGI app in-process - no live socket, no honest-skip
needed for the HTTP layer itself.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "packages"))

from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from local_ai_core.deployment.app_context import AppContext, build_app_context  # noqa: E402
from local_ai_core.deployment.config import AppConfig, load_config  # noqa: E402
from local_ai_core.deployment.health import (  # noqa: E402
    run_liveness_check,
    run_readiness_check,
)
from local_ai_core.gateway.queue import QueueFullError  # noqa: E402
from local_ai_core.runtimes.types import LLMRequest  # noqa: E402
from local_ai_core.security.guard_pipeline import GuardVerdict  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "app.example.yaml"
DEFAULT_CATALOG_PATH = REPO_ROOT / "models" / "MODEL_CATALOG.md"


def _build_default_context() -> AppContext:
    config_path = os.environ.get("APP_CONFIG_PATH", str(DEFAULT_CONFIG_PATH))
    config: AppConfig = load_config(config_path)
    return build_app_context(config, model_catalog_path=DEFAULT_CATALOG_PATH)


app = FastAPI(title="Local AI App (Module 23)")


def get_ctx() -> AppContext:
    """Lazily builds (and caches) the default `AppContext` on first use -
    never at import time, so tests can inject a test-scoped context via
    `app.state.ctx` before any request triggers the real default (which
    would otherwise touch the real `~/.local-llm-ai` data directory).
    """
    if getattr(app.state, "ctx", None) is None:
        app.state.ctx = _build_default_context()
    return app.state.ctx


class ChatRequest(BaseModel):
    prompt: str
    model: str | None = None


class ChatResponse(BaseModel):
    text: str
    model: str


@app.get("/health")
def health() -> dict:
    result = run_liveness_check()
    return {"status": "ok" if result.passed else "fail", "detail": result.detail}


@app.get("/ready")
def ready() -> dict:
    ctx: AppContext = get_ctx()
    result = run_readiness_check(ctx.data_dir, ctx.model_registry)
    if not result.passed:
        raise HTTPException(status_code=503, detail=result.detail)
    return {"status": "ready", "detail": result.detail}


@app.get("/models")
def models() -> list[dict]:
    ctx: AppContext = get_ctx()
    return [
        {"model_id": e.model_id, "category": e.category, "recommended_ram_tier": e.recommended_ram_tier}
        for e in ctx.model_registry.all_entries()
    ]


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    ctx: AppContext = get_ctx()

    decision = ctx.guard_classifier.classify(request.prompt)
    if decision.verdict == GuardVerdict.BLOCK:
        raise HTTPException(status_code=400, detail=f"request blocked: {decision.reason}")

    model = request.model or ctx.config.models.default_chat

    async def _generate() -> str:
        response = await ctx.runtime.generate(LLMRequest(model=model, prompt=request.prompt))
        return response.text

    try:
        queued_result = await ctx.admission_controller.submit(_generate)
    except QueueFullError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc

    return ChatResponse(text=queued_result.result, model=model)

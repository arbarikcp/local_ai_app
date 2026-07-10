"""ExtractionAppContext — the composition root for this project, extending
(not replacing) Module 23's `AppContext` with a `storage` handle
(ARCHITECTURE.md "High-level", "Deployment shape"). `run_extraction()` is
the one function that ties normalization, the reused Module 8 pipeline,
transport-level retry, and persistence together - the FastAPI layer calls
this and nothing else.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path

from local_ai_core.deployment.app_context import AppContext, build_app_context
from local_ai_core.deployment.config import AppConfig
from local_ai_core.extraction.pipeline import ExtractionPipeline
from local_ai_core.runtimes.base import LLMRuntime, Timer, with_retries
from local_ai_core.runtimes.errors import RequestTimeout, RuntimeUnavailable

from extraction_normalization import normalize_text
from extraction_prompts import resolve_schema
from extraction_storage import ExtractionRecord, ExtractionStore

MAX_REPAIR_ATTEMPTS = 2  # curriculum's own cap for this project ("at most 2 times")


@dataclass
class ExtractionAppContext:
    base: AppContext
    storage: ExtractionStore


def build_extraction_context(
    config: AppConfig,
    *,
    model_catalog_path: str | Path,
    runtime: LLMRuntime | None = None,
) -> ExtractionAppContext:
    base = build_app_context(config, model_catalog_path=model_catalog_path, runtime=runtime)
    extraction_db_path = base.data_dir.base_dir / "extraction" / "extraction.db"
    extraction_db_path.parent.mkdir(parents=True, exist_ok=True)
    storage = ExtractionStore(extraction_db_path)
    return ExtractionAppContext(base=base, storage=storage)


async def run_extraction(
    ctx: ExtractionAppContext,
    *,
    schema_name: str,
    text: str,
    model: str | None = None,
    max_input_chars: int | None = None,
) -> ExtractionRecord:
    """Raises `SchemaNotFoundError` (extraction_prompts.py) for an unknown
    schema, `TextTooLongError` (extraction_normalization.py) for
    over-length input - both before any LLM call - and propagates
    `RuntimeUnavailable`/`RequestTimeout` (Module 6) after `with_retries`
    exhausts its attempts.
    """
    registered = resolve_schema(schema_name)
    normalized = normalize_text(text, max_chars=max_input_chars)
    resolved_model = model or ctx.base.config.models.default_extraction

    pipeline: ExtractionPipeline = ExtractionPipeline(
        ctx.base.runtime, registered.schema_class, max_repair_attempts=MAX_REPAIR_ATTEMPTS
    )

    async def _run_pipeline():
        return await pipeline.run(normalized, resolved_model)

    timer = Timer()
    result = await with_retries(_run_pipeline, retryable=(RuntimeUnavailable, RequestTimeout))
    latency_ms = timer.elapsed_ms

    trace_id = str(uuid.uuid4())
    record = ExtractionRecord(
        request_id=str(uuid.uuid4()),
        trace_id=trace_id,
        schema_name=schema_name,
        raw_input=text,
        extracted_fields=result.fields,
        confidence=result.confidence,
        needs_review=result.needs_review,
        validation_error=result.validation_error,
        used_repair_retry=result.used_repair_retry,
        latency_ms=latency_ms,
    )
    ctx.storage.save(record)

    ctx.base.audit_log.record(
        trace_id,
        "extraction",
        {"schema_name": schema_name},
        "needs_review" if result.needs_review else "success",
        result.validation_error or "",
    )

    return record

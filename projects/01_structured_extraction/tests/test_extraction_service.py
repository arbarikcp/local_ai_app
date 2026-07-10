import pytest

from extraction_normalization import TextTooLongError
from extraction_prompts import SchemaNotFoundError
from extraction_service import build_extraction_context, run_extraction
from local_ai_core.deployment.config import AppConfig
from local_ai_core.runtimes.fake import FakeRuntime

REPO_ROOT_CATALOG = "models/MODEL_CATALOG.md"


class SequencedRuntime(FakeRuntime):
    """Returns responses from a fixed sequence, one per call (same pattern
    Module 8's own pipeline tests use for exercising the repair-retry path).
    """

    def __init__(self, responses: list[str], **kwargs):
        super().__init__(**kwargs)
        self._sequence = iter(responses)

    async def generate(self, request):
        text = next(self._sequence)
        self.responses = {request.model: text}
        return await super().generate(request)


def make_config(tmp_path) -> AppConfig:
    return AppConfig.model_validate(
        {
            "app": {"data_dir": str(tmp_path / "data")},
            "models": {
                "default_chat": "a",
                "default_extraction": "test-extraction-model",
                "default_code": "c",
                "default_embedding": "d",
            },
        }
    )


class TestBuildExtractionContext:
    def test_wires_a_storage_handle_under_the_data_dir(self, tmp_path):
        config = make_config(tmp_path)
        ctx = build_extraction_context(config, model_catalog_path=REPO_ROOT_CATALOG)
        assert ctx.storage is not None
        assert (ctx.base.data_dir.base_dir / "extraction" / "extraction.db").exists()


class TestRunExtractionHappyPath:
    async def test_a_valid_response_is_extracted_and_persisted(self, tmp_path):
        config = make_config(tmp_path)
        runtime = FakeRuntime(default_response='{"invoice_number": "A-1", "vendor_name": "Acme", '
        '"invoice_date": "2026-01-01", "currency": "USD", "total_amount": 10.0, '
        '"confidence": "high", "evidence": {}}')
        ctx = build_extraction_context(config, model_catalog_path=REPO_ROOT_CATALOG, runtime=runtime)

        record = await run_extraction(ctx, schema_name="invoice_v1", text="Invoice #A-1 for $10 from Acme.")

        assert record.schema_name == "invoice_v1"
        assert record.extracted_fields["invoice_number"] == "A-1"
        assert record.needs_review is False
        assert record.latency_ms >= 0

        stored = ctx.storage.get(record.request_id)
        assert stored is not None
        assert stored.raw_input == "Invoice #A-1 for $10 from Acme."


class TestRunExtractionRepairRetry:
    async def test_an_invalid_first_response_is_repaired_and_flagged(self, tmp_path):
        config = make_config(tmp_path)
        runtime = SequencedRuntime(
            [
                "not valid json at all",
                '{"invoice_number": "A-1", "vendor_name": "Acme", "invoice_date": null, '
                '"currency": "USD", "total_amount": 10.0, "confidence": "medium", "evidence": {}}',
            ]
        )
        ctx = build_extraction_context(config, model_catalog_path=REPO_ROOT_CATALOG, runtime=runtime)

        record = await run_extraction(ctx, schema_name="invoice_v1", text="Invoice #A-1.")

        assert record.used_repair_retry is True
        assert record.extracted_fields["invoice_number"] == "A-1"

    async def test_repair_is_capped_at_two_attempts(self, tmp_path):
        config = make_config(tmp_path)
        call_count = {"n": 0}

        class CountingRuntime(FakeRuntime):
            async def generate(self, request):
                call_count["n"] += 1
                return await super().generate(request)

        runtime = CountingRuntime(default_response="still not valid json")
        ctx = build_extraction_context(config, model_catalog_path=REPO_ROOT_CATALOG, runtime=runtime)

        record = await run_extraction(ctx, schema_name="invoice_v1", text="Invoice #A-1.")

        assert call_count["n"] == 3  # 1 initial + 2 repair attempts (this project's cap)
        assert record.needs_review is True


class TestRunExtractionValidation:
    async def test_unknown_schema_name_raises_before_any_llm_call(self, tmp_path):
        config = make_config(tmp_path)
        call_count = {"n": 0}

        class CountingRuntime(FakeRuntime):
            async def generate(self, request):
                call_count["n"] += 1
                return await super().generate(request)

        ctx = build_extraction_context(config, model_catalog_path=REPO_ROOT_CATALOG, runtime=CountingRuntime())

        with pytest.raises(SchemaNotFoundError):
            await run_extraction(ctx, schema_name="does_not_exist_v1", text="hello")
        assert call_count["n"] == 0

    async def test_over_length_text_raises_before_any_llm_call(self, tmp_path):
        config = make_config(tmp_path)
        call_count = {"n": 0}

        class CountingRuntime(FakeRuntime):
            async def generate(self, request):
                call_count["n"] += 1
                return await super().generate(request)

        ctx = build_extraction_context(config, model_catalog_path=REPO_ROOT_CATALOG, runtime=CountingRuntime())

        with pytest.raises(TextTooLongError):
            await run_extraction(ctx, schema_name="invoice_v1", text="a" * 1000, max_input_chars=100)
        assert call_count["n"] == 0


class TestRunExtractionAuditLogging:
    async def test_every_extraction_is_audit_logged(self, tmp_path):
        config = make_config(tmp_path)
        runtime = FakeRuntime(default_response='{"invoice_number": "A-1", "confidence": "high", "evidence": {}}')
        ctx = build_extraction_context(config, model_catalog_path=REPO_ROOT_CATALOG, runtime=runtime)

        record = await run_extraction(ctx, schema_name="invoice_v1", text="Invoice #A-1.")

        entries = ctx.base.audit_log.entries_for_trace(record.trace_id)
        assert len(entries) == 1
        assert entries[0].tool_name == "extraction"

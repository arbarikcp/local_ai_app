import json
from pathlib import Path

import pytest
from local_ai_core.deployment.config import AppConfig
from local_ai_core.multimodal.vlm import FakeVLM
from local_ai_core.runtimes.fake import FakeRuntime

from doc_service import build_doc_context, run_extract, run_ingest, run_query

REPO_ROOT_CATALOG = "models/MODEL_CATALOG.md"
FIXTURE_PATH = Path(__file__).resolve().parent.parent.parent.parent / "datasets" / "multimodal" / "project_04" / "multi_page_form.pdf"

VALID_RESPONSE = json.dumps(
    {
        "document_type": "account_closure_request",
        "applicant_name": "Jordan Rivera",
        "key_date": "2026-06-15",
        "key_amount": 42.50,
        "notes": None,
        "confidence": "high",
        "evidence": {},
    }
)


def make_config(tmp_path) -> AppConfig:
    return AppConfig.model_validate(
        {
            "app": {"data_dir": str(tmp_path / "data")},
            "models": {
                "default_chat": "chat-model",
                "default_extraction": "extraction-model",
                "default_code": "code-model",
                "default_embedding": "embedding-model",
            },
        }
    )


class TestBuildDocContext:
    def test_wires_a_real_persistent_store_and_defaults_to_fake_vlm(self, tmp_path):
        config = make_config(tmp_path)
        ctx = build_doc_context(config, model_catalog_path=REPO_ROOT_CATALOG)

        assert ctx.storage is not None
        assert isinstance(ctx.vlm, FakeVLM)
        assert (ctx.base.data_dir.base_dir / "multimodal" / "multimodal.db").exists()

    def test_accepts_an_injected_vlm(self, tmp_path):
        config = make_config(tmp_path)
        custom_vlm = FakeVLM(default_response="custom description")
        ctx = build_doc_context(config, model_catalog_path=REPO_ROOT_CATALOG, vlm=custom_vlm)
        assert ctx.vlm is custom_vlm


@pytest.mark.asyncio
class TestFullIngestExtractQueryRoundTrip:
    async def test_ingest_then_query_through_the_composition_root(self, tmp_path):
        config = make_config(tmp_path)
        runtime = FakeRuntime(
            responses={"chat-model": "The refund amount owed is $42.50 [multi_page_form::page2]."},
            default_response=VALID_RESPONSE,
        )
        ctx = build_doc_context(config, model_catalog_path=REPO_ROOT_CATALOG, runtime=runtime)

        ingestion_result = await run_ingest(ctx, FIXTURE_PATH)
        assert ingestion_result.page_count == 3

        stored_doc = ctx.storage.get_document("multi_page_form")
        assert stored_doc.status == "ingested"

        qa_result = await run_query(ctx, "multi_page_form", "What is the refund amount?")
        assert qa_result.citations[0].page_id == "multi_page_form::page2"
        assert qa_result.citations[0].verified is True

    async def test_run_extract_re_runs_extraction_for_a_single_page(self, tmp_path):
        config = make_config(tmp_path)
        runtime = FakeRuntime(default_response=VALID_RESPONSE)
        ctx = build_doc_context(config, model_catalog_path=REPO_ROOT_CATALOG, runtime=runtime)

        await run_ingest(ctx, FIXTURE_PATH)
        updated = await run_extract(ctx, "multi_page_form", page_number=1)

        assert len(updated) == 1
        assert updated[0].page_number == 1
        assert updated[0].extracted_fields["applicant_name"] == "Jordan Rivera"

    async def test_run_extract_skips_the_vlm_routed_page(self, tmp_path):
        config = make_config(tmp_path)
        runtime = FakeRuntime(default_response=VALID_RESPONSE)
        ctx = build_doc_context(config, model_catalog_path=REPO_ROOT_CATALOG, runtime=runtime)

        await run_ingest(ctx, FIXTURE_PATH)
        updated = await run_extract(ctx, "multi_page_form")

        assert {page.page_number for page in updated} == {1, 2}

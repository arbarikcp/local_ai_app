import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import doc_api as sut
from doc_service import build_doc_context
from local_ai_core.deployment.config import AppConfig
from local_ai_core.runtimes.fake import FakeRuntime

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


def make_test_context(tmp_path, *, default_response: str = "answer", responses: dict | None = None):
    config = AppConfig.model_validate(
        {
            "app": {"data_dir": str(tmp_path / "data")},
            "models": {
                "default_chat": "test-chat-model",
                "default_extraction": "test-extraction-model",
                "default_code": "c",
                "default_embedding": "d",
            },
        }
    )
    runtime = FakeRuntime(default_response=default_response, responses=responses or {})
    return build_doc_context(config, model_catalog_path=REPO_ROOT_CATALOG, runtime=runtime)


@pytest.fixture
def client(tmp_path):
    sut.app.state.ctx = make_test_context(tmp_path, default_response=VALID_RESPONSE)
    return TestClient(sut.app)


class TestHealthAndReady:
    def test_health_reports_ok(self, client):
        assert client.get("/health").json()["status"] == "ok"

    def test_ready_reports_ready(self, client):
        assert client.get("/ready").json()["status"] == "ready"


class TestIngestDocument:
    def test_ingesting_the_real_fixture_routes_pages_correctly(self, client):
        response = client.post("/documents", json={"source_path": str(FIXTURE_PATH)})
        assert response.status_code == 200
        body = response.json()
        assert len(body) == 3
        assert [page["route"] for page in body] == ["text_llm", "text_llm", "vlm"]

    def test_missing_file_returns_404(self, client):
        response = client.post("/documents", json={"source_path": "/nonexistent/file.pdf"})
        assert response.status_code == 404

    def test_non_pdf_file_returns_422(self, tmp_path, client):
        not_a_pdf = tmp_path / "notes.txt"
        not_a_pdf.write_text("hello")
        response = client.post("/documents", json={"source_path": str(not_a_pdf)})
        assert response.status_code == 422


class TestGetDocument:
    def test_a_stored_document_can_be_retrieved_with_its_pages(self, client):
        client.post("/documents", json={"source_path": str(FIXTURE_PATH)})
        response = client.get("/documents/multi_page_form")
        assert response.status_code == 200
        body = response.json()
        assert body["page_count"] == 3
        assert len(body["pages"]) == 3

    def test_unknown_doc_id_returns_404(self, client):
        response = client.get("/documents/does-not-exist")
        assert response.status_code == 404


class TestExtract:
    def test_extracting_a_single_page_returns_structured_fields(self, client):
        client.post("/documents", json={"source_path": str(FIXTURE_PATH)})
        response = client.post("/documents/multi_page_form/extract", json={"page_number": 1})
        assert response.status_code == 200
        body = response.json()
        assert len(body) == 1
        assert body[0]["extracted_fields"]["applicant_name"] == "Jordan Rivera"

    def test_extracting_all_pages_skips_the_vlm_route(self, client):
        client.post("/documents", json={"source_path": str(FIXTURE_PATH)})
        response = client.post("/documents/multi_page_form/extract", json={})
        assert response.status_code == 200
        assert len(response.json()) == 2

    def test_unknown_doc_id_returns_404(self, client):
        response = client.post("/documents/does-not-exist/extract", json={})
        assert response.status_code == 404


class TestQuery:
    def test_a_grounded_answer_returns_verified_citations(self, tmp_path):
        sut.app.state.ctx = make_test_context(
            tmp_path,
            default_response=VALID_RESPONSE,
            responses={
                "test-chat-model": "The refund amount owed is $42.50 [multi_page_form::page2]."
            },
        )
        client = TestClient(sut.app)
        client.post("/documents", json={"source_path": str(FIXTURE_PATH)})

        response = client.post("/documents/multi_page_form/query", json={"question": "What is the refund amount?"})
        assert response.status_code == 200
        body = response.json()
        assert body["citations"][0]["verified"] is True
        assert body["citations"][0]["page_number"] == 2

    def test_unknown_doc_id_returns_404(self, client):
        response = client.post("/documents/does-not-exist/query", json={"question": "anything?"})
        assert response.status_code == 404

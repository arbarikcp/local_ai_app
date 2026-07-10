import pytest
from fastapi.testclient import TestClient

import extraction_api as sut
from extraction_service import build_extraction_context
from local_ai_core.deployment.config import AppConfig
from local_ai_core.runtimes.fake import FakeRuntime

REPO_ROOT_CATALOG = "models/MODEL_CATALOG.md"

VALID_INVOICE_RESPONSE = (
    '{"invoice_number": "A-1", "vendor_name": "Acme", "invoice_date": "2026-01-01", '
    '"currency": "USD", "total_amount": 10.0, "confidence": "high", "evidence": {}}'
)


def make_test_context(tmp_path, *, default_response: str = VALID_INVOICE_RESPONSE):
    config = AppConfig.model_validate(
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
    runtime = FakeRuntime(default_response=default_response)
    return build_extraction_context(config, model_catalog_path=REPO_ROOT_CATALOG, runtime=runtime)


@pytest.fixture
def client(tmp_path):
    sut.app.state.ctx = make_test_context(tmp_path)
    return TestClient(sut.app)


class TestHealthAndReady:
    def test_health_reports_ok(self, client):
        assert client.get("/health").json()["status"] == "ok"

    def test_ready_reports_ready(self, client):
        assert client.get("/ready").json()["status"] == "ready"


class TestExtractHappyPath:
    def test_a_valid_invoice_extraction_succeeds(self, client):
        response = client.post("/extract", json={"schema_name": "invoice_v1", "text": "Invoice #A-1."})
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "success"
        assert body["data"]["invoice_number"] == "A-1"
        assert body["confidence"] == "high"
        assert body["latency_ms"] >= 0

    def test_a_second_schema_also_works(self, tmp_path):
        sut.app.state.ctx = make_test_context(
            tmp_path, default_response='{"category": "billing", "urgency": "high"}'
        )
        client = TestClient(sut.app)
        response = client.post("/extract", json={"schema_name": "support_ticket_v1", "text": "Charged twice."})
        assert response.status_code == 200
        assert response.json()["data"]["category"] == "billing"


class TestExtractErrors:
    def test_unknown_schema_name_returns_404(self, client):
        response = client.post("/extract", json={"schema_name": "does_not_exist_v1", "text": "hello"})
        assert response.status_code == 404

    def test_over_length_text_returns_413(self, client):
        long_text = "a" * 1_000_000
        response = client.post("/extract", json={"schema_name": "invoice_v1", "text": long_text})
        assert response.status_code == 413


class TestGetExtraction:
    def test_a_stored_extraction_can_be_retrieved_by_request_id(self, client):
        create_response = client.post("/extract", json={"schema_name": "invoice_v1", "text": "Invoice #A-1."})
        request_id = create_response.json()["request_id"]

        get_response = client.get(f"/extractions/{request_id}")
        assert get_response.status_code == 200
        assert get_response.json()["request_id"] == request_id

    def test_unknown_request_id_returns_404(self, client):
        response = client.get("/extractions/does-not-exist")
        assert response.status_code == 404


class TestLowConfidence:
    def test_a_needs_review_extraction_appears_in_the_low_confidence_list(self, tmp_path):
        sut.app.state.ctx = make_test_context(tmp_path, default_response="not valid json")
        client = TestClient(sut.app)

        client.post("/extract", json={"schema_name": "invoice_v1", "text": "Invoice #A-1."})

        response = client.get("/extractions/low-confidence")
        assert response.status_code == 200
        body = response.json()
        assert len(body) == 1
        assert body[0]["status"] == "needs_review"

    def test_a_successful_extraction_does_not_appear(self, client):
        client.post("/extract", json={"schema_name": "invoice_v1", "text": "Invoice #A-1."})
        response = client.get("/extractions/low-confidence")
        assert response.json() == []

    def test_route_ordering_does_not_treat_low_confidence_as_a_request_id(self, client):
        # /extractions/low-confidence must not be shadowed by
        # /extractions/{request_id}.
        response = client.get("/extractions/low-confidence")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

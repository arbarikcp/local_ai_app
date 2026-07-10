from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import gw_api as sut
from gw_service import build_gw_context
from local_ai_core.deployment.config import AppConfig
from local_ai_core.runtimes.errors import RuntimeUnavailable
from local_ai_core.runtimes.fake import FakeRuntime

REPO_ROOT_CATALOG = "models/MODEL_CATALOG.md"
ROUTES_PATH = Path(__file__).resolve().parent.parent / "config" / "gateway_routes.yaml"


def make_test_context(tmp_path, *, runtime=None, fallback_runtime=None, timeout_seconds: float = 30.0):
    config = AppConfig.model_validate(
        {
            "app": {"data_dir": str(tmp_path / "data")},
            "models": {
                "default_chat": "llama3.1:8b-instruct",
                "default_extraction": "qwen2.5:7b-instruct",
                "default_code": "qwen2.5-coder:7b",
                "default_embedding": "nomic-embed-text",
            },
        }
    )
    return build_gw_context(
        config, model_catalog_path=REPO_ROOT_CATALOG, routes_path=ROUTES_PATH,
        runtime=runtime or FakeRuntime(responses={"llama3.1:8b-instruct": "hi there"}),
        fallback_runtime=fallback_runtime, timeout_seconds=timeout_seconds,
    )


@pytest.fixture
def client(tmp_path):
    sut.app.state.ctx = make_test_context(tmp_path)
    return TestClient(sut.app)


class TestHealthAndReady:
    def test_health_reports_ok(self, client):
        assert client.get("/health").json()["status"] == "ok"

    def test_ready_reports_ready(self, client):
        assert client.get("/ready").json()["status"] == "ready"


class TestGenerate:
    def test_a_real_request_returns_a_real_answer(self, client):
        response = client.post("/generate", json={"task": "chat", "prompt": "hello"})
        assert response.status_code == 200
        body = response.json()
        assert body["answer"] == "hi there"
        assert body["model_used"] == "llama3.1:8b-instruct"
        assert body["used_fallback"] is False
        assert body["trace_id"]

    def test_an_unknown_task_returns_404(self, client):
        response = client.post("/generate", json={"task": "does-not-exist", "prompt": "hi"})
        assert response.status_code == 404

    def test_a_failing_primary_returns_a_fallback_answer(self, tmp_path):
        sut.app.state.ctx = make_test_context(
            tmp_path,
            runtime=FakeRuntime(fail_with=RuntimeUnavailable("primary down")),
            fallback_runtime=FakeRuntime(responses={"qwen2.5:1.5b-instruct": "fallback answer"}),
        )
        client = TestClient(sut.app)
        response = client.post("/generate", json={"task": "chat", "prompt": "hello"})
        assert response.status_code == 200
        body = response.json()
        assert body["used_fallback"] is True
        assert body["model_used"] == "qwen2.5:1.5b-instruct"

    def test_both_models_failing_returns_503(self, tmp_path):
        sut.app.state.ctx = make_test_context(
            tmp_path,
            runtime=FakeRuntime(fail_with=RuntimeUnavailable("primary down")),
            fallback_runtime=FakeRuntime(fail_with=RuntimeUnavailable("fallback down")),
        )
        client = TestClient(sut.app)
        response = client.post("/generate", json={"task": "chat", "prompt": "hello"})
        assert response.status_code == 503

    def test_a_slow_request_beyond_timeout_returns_504(self, tmp_path):
        sut.app.state.ctx = make_test_context(
            tmp_path, runtime=FakeRuntime(simulated_latency_ms=50), timeout_seconds=0.01
        )
        client = TestClient(sut.app)
        response = client.post("/generate", json={"task": "chat", "prompt": "hello"})
        assert response.status_code == 504


class TestStream:
    def test_a_real_stream_delivers_chunks(self, client):
        with client.stream("POST", "/stream", json={"task": "chat", "prompt": "hello"}) as response:
            assert response.status_code == 200
            text = "".join(response.iter_text())
        assert "hi there" in text

    def test_an_unknown_task_returns_404_before_streaming(self, client):
        response = client.post("/stream", json={"task": "does-not-exist", "prompt": "hi"})
        assert response.status_code == 404


class TestGetRequest:
    def test_a_completed_request_is_retrievable_by_request_id(self, client):
        client.post("/generate", json={"task": "chat", "prompt": "hello"})
        stored = sut.app.state.ctx.storage.list_requests(task="chat")
        assert len(stored) == 1

        response = client.get(f"/requests/{stored[0].request_id}")
        assert response.status_code == 200
        assert response.json()["task"] == "chat"

    def test_unknown_request_id_returns_404(self, client):
        response = client.get("/requests/does-not-exist")
        assert response.status_code == 404


class TestBenchmark:
    def test_benchmarking_a_single_task_returns_two_rows(self, client):
        response = client.post("/benchmark", json={"task": "chat", "repeats": 1})
        assert response.status_code == 200
        names = {row["name"] for row in response.json()}
        assert names == {"chat-primary", "chat-fallback"}

    def test_benchmarking_an_unknown_task_returns_404(self, client):
        response = client.post("/benchmark", json={"task": "does-not-exist"})
        assert response.status_code == 404

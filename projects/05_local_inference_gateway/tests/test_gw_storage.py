import pytest

from gw_storage import GatewayRequestRecord, GwStorage


@pytest.fixture
def storage(tmp_path):
    with GwStorage(tmp_path / "gateway.db") as store:
        yield store


class TestGatewayRequestPersistence:
    def test_save_and_get_round_trips(self, storage):
        storage.save_request(
            GatewayRequestRecord(
                request_id="req-1",
                trace_id="trace-1",
                task="chat",
                model_used="llama3.1:8b-instruct",
                used_fallback=False,
                latency_ms=12.5,
                status="ok",
            )
        )
        record = storage.get_request("req-1")
        assert record is not None
        assert record.task == "chat"
        assert record.used_fallback is False
        assert record.created_at is not None

    def test_get_returns_none_when_missing(self, storage):
        assert storage.get_request("nonexistent") is None

    def test_save_rejects_invalid_status(self, storage):
        with pytest.raises(ValueError):
            storage.save_request(
                GatewayRequestRecord(
                    request_id="req-1",
                    trace_id="trace-1",
                    task="chat",
                    model_used="m",
                    used_fallback=False,
                    latency_ms=1.0,
                    status="bogus",
                )
            )

    def test_list_requests_filters_by_task(self, storage):
        storage.save_request(
            GatewayRequestRecord(
                request_id="req-1", trace_id="t1", task="chat", model_used="m", used_fallback=False,
                latency_ms=1.0, status="ok",
            )
        )
        storage.save_request(
            GatewayRequestRecord(
                request_id="req-2", trace_id="t2", task="code", model_used="m", used_fallback=True,
                latency_ms=2.0, status="ok",
            )
        )
        chat_only = storage.list_requests(task="chat")
        assert [r.request_id for r in chat_only] == ["req-1"]

    def test_list_requests_without_filter_returns_all(self, storage):
        storage.save_request(
            GatewayRequestRecord(
                request_id="req-1", trace_id="t1", task="chat", model_used="m", used_fallback=False,
                latency_ms=1.0, status="ok",
            )
        )
        storage.save_request(
            GatewayRequestRecord(
                request_id="req-2", trace_id="t2", task="code", model_used="m", used_fallback=True,
                latency_ms=2.0, status="timeout",
            )
        )
        assert len(storage.list_requests()) == 2

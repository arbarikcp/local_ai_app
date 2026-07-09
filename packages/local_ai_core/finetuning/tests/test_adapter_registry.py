from local_ai_core.finetuning.adapter_registry import AdapterRecord, AdapterRegistry


def make_record(name: str = "ticket-classifier-v1") -> AdapterRecord:
    return AdapterRecord(
        name=name,
        base_model="mlx-community/Qwen2.5-1.5B-Instruct-4bit",
        rank=8,
        alpha=16,
        target_modules=["q_proj", "v_proj"],
        dataset_hash="abc123",
        file_path="adapters/ticket-classifier-v1",
    )


class TestRegisterAndGet:
    def test_registered_adapter_can_be_retrieved(self):
        with AdapterRegistry() as registry:
            registry.register(make_record())
            record = registry.get("ticket-classifier-v1")
            assert record is not None
            assert record.base_model == "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
            assert record.rank == 8
            assert record.alpha == 16
            assert record.target_modules == ["q_proj", "v_proj"]
            assert record.dataset_hash == "abc123"
            assert record.created_at is not None

    def test_missing_adapter_returns_none(self):
        with AdapterRegistry() as registry:
            assert registry.get("does-not-exist") is None

    def test_duplicate_name_raises(self):
        import sqlite3

        with AdapterRegistry() as registry:
            registry.register(make_record())
            try:
                registry.register(make_record())
                assert False, "expected sqlite3.IntegrityError"
            except sqlite3.IntegrityError:
                pass


class TestListAndDelete:
    def test_list_adapters_returns_every_registered_record(self):
        with AdapterRegistry() as registry:
            registry.register(make_record("adapter-a"))
            registry.register(make_record("adapter-b"))
            names = {record.name for record in registry.list_adapters()}
            assert names == {"adapter-a", "adapter-b"}

    def test_delete_removes_the_adapter(self):
        with AdapterRegistry() as registry:
            registry.register(make_record())
            registry.delete("ticket-classifier-v1")
            assert registry.get("ticket-classifier-v1") is None


class TestPersistsAcrossCloseAndReopen:
    def test_adapter_survives_a_real_close_and_reopen_cycle(self, tmp_path):
        db_path = tmp_path / "adapters.db"

        registry = AdapterRegistry(db_path)
        registry.register(make_record())
        registry.close()

        reopened = AdapterRegistry(db_path)
        record = reopened.get("ticket-classifier-v1")
        reopened.close()

        assert record is not None
        assert record.file_path == "adapters/ticket-classifier-v1"

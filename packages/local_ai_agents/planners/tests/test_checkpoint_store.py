from local_ai_agents.planners.checkpoint_store import CheckpointStore


class TestSaveAndLoad:
    def test_saved_checkpoint_is_retrievable(self, tmp_path):
        store = CheckpointStore(tmp_path / "checkpoints.db")
        store.save("run-1", "gather", {"tickets_found": 3}, step_index=2)
        checkpoint = store.load("run-1")
        assert checkpoint is not None
        assert checkpoint.node_name == "gather"
        assert checkpoint.state == {"tickets_found": 3}
        assert checkpoint.step_index == 2
        store.close()

    def test_missing_run_id_returns_none(self, tmp_path):
        store = CheckpointStore(tmp_path / "checkpoints.db")
        assert store.load("does-not-exist") is None
        store.close()

    def test_saving_again_for_the_same_run_id_overwrites(self, tmp_path):
        store = CheckpointStore(tmp_path / "checkpoints.db")
        store.save("run-1", "classify", {}, step_index=0)
        store.save("run-1", "gather", {"progress": "half"}, step_index=1)
        checkpoint = store.load("run-1")
        assert checkpoint.node_name == "gather"
        assert checkpoint.step_index == 1
        store.close()


class TestDelete:
    def test_removes_the_checkpoint(self, tmp_path):
        store = CheckpointStore(tmp_path / "checkpoints.db")
        store.save("run-1", "classify", {}, step_index=0)
        store.delete("run-1")
        assert store.load("run-1") is None
        store.close()

    def test_deleting_a_missing_run_id_is_not_an_error(self, tmp_path):
        store = CheckpointStore(tmp_path / "checkpoints.db")
        store.delete("does-not-exist")
        store.close()


class TestPersistenceAcrossRestart:
    def test_checkpoint_survives_a_close_and_reopen(self, tmp_path):
        db_path = tmp_path / "checkpoints.db"
        store1 = CheckpointStore(db_path)
        store1.save("run-1", "gather", {"tickets_found": 3}, step_index=2)
        store1.close()

        store2 = CheckpointStore(db_path)
        checkpoint = store2.load("run-1")
        assert checkpoint is not None
        assert checkpoint.node_name == "gather"
        store2.close()

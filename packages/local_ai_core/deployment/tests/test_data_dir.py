from local_ai_core.deployment.config import AppConfig
from local_ai_core.deployment.data_dir import ensure_data_dir_layout, resolve_data_dir


def make_config(data_dir: str) -> AppConfig:
    return AppConfig.model_validate(
        {
            "app": {"data_dir": data_dir},
            "models": {
                "default_chat": "a",
                "default_extraction": "b",
                "default_code": "c",
                "default_embedding": "d",
            },
        }
    )


class TestResolveDataDir:
    def test_expands_the_home_directory_shorthand(self):
        config = make_config("~/.local-llm-ai-test")
        resolved = resolve_data_dir(config)
        assert "~" not in str(resolved)
        assert resolved.is_absolute()


class TestEnsureDataDirLayout:
    def test_creates_every_real_subdirectory(self, tmp_path):
        config = make_config(str(tmp_path / "data"))
        layout = ensure_data_dir_layout(config)

        assert layout.sessions_db.parent.is_dir()
        assert layout.audit_db.parent.is_dir()
        assert layout.adapters_db.parent.is_dir()
        assert layout.eval_feedback_db.parent.is_dir()
        assert layout.backups_dir.is_dir()

    def test_is_idempotent(self, tmp_path):
        config = make_config(str(tmp_path / "data"))
        ensure_data_dir_layout(config)
        layout = ensure_data_dir_layout(config)  # should not raise
        assert layout.base_dir.is_dir()

    def test_db_paths_are_distinct_files_under_the_base_dir(self, tmp_path):
        config = make_config(str(tmp_path / "data"))
        layout = ensure_data_dir_layout(config)
        paths = {layout.sessions_db, layout.audit_db, layout.adapters_db, layout.eval_feedback_db}
        assert len(paths) == 4
        assert all(str(p).startswith(str(layout.base_dir)) for p in paths)

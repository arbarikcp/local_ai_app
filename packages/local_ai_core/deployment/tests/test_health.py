from local_ai_core.deployment.config import AppConfig
from local_ai_core.deployment.data_dir import ensure_data_dir_layout
from local_ai_core.deployment.health import (
    run_liveness_check,
    run_readiness_check,
    run_startup_checks,
)
from local_ai_core.deployment.model_registry import ModelRegistry, load_model_registry

REPO_ROOT_CATALOG = "models/MODEL_CATALOG.md"


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


class TestRunStartupChecks:
    def test_all_checks_pass_against_a_real_writable_dir_and_real_catalog(self, tmp_path):
        config = make_config(str(tmp_path / "data"))
        results = run_startup_checks(config, model_catalog_path=REPO_ROOT_CATALOG)
        assert all(r.passed for r in results)
        assert {r.name for r in results} == {
            "config_valid",
            "data_dir_writable",
            "model_catalog_parseable",
            "disk_space",
        }

    def test_a_missing_model_catalog_fails_that_check_only(self, tmp_path):
        config = make_config(str(tmp_path / "data"))
        results = run_startup_checks(config, model_catalog_path=tmp_path / "does_not_exist.md")
        by_name = {r.name: r for r in results}
        assert by_name["model_catalog_parseable"].passed is False
        assert by_name["data_dir_writable"].passed is True

    def test_model_catalog_check_reports_the_real_entry_count(self, tmp_path):
        config = make_config(str(tmp_path / "data"))
        results = run_startup_checks(config, model_catalog_path=REPO_ROOT_CATALOG)
        by_name = {r.name: r for r in results}
        assert "10 entries" in by_name["model_catalog_parseable"].detail


class TestRunReadinessCheck:
    def test_ready_when_data_dir_exists_and_registry_is_populated(self, tmp_path):
        config = make_config(str(tmp_path / "data"))
        layout = ensure_data_dir_layout(config)
        registry = load_model_registry(REPO_ROOT_CATALOG)
        result = run_readiness_check(layout, registry)
        assert result.passed is True

    def test_not_ready_when_registry_is_empty(self, tmp_path):
        config = make_config(str(tmp_path / "data"))
        layout = ensure_data_dir_layout(config)
        empty_registry = ModelRegistry([])
        result = run_readiness_check(layout, empty_registry)
        assert result.passed is False
        assert "empty" in result.detail


class TestRunLivenessCheck:
    def test_always_passes(self):
        result = run_liveness_check()
        assert result.passed is True

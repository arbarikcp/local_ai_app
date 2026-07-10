from local_ai_core.deployment.app_context import build_app_context
from local_ai_core.deployment.config import AppConfig
from local_ai_core.runtimes.fake import FakeRuntime
from local_ai_core.runtimes.types import LLMRequest

REPO_ROOT_CATALOG = "models/MODEL_CATALOG.md"


def make_config(data_dir: str, **overrides) -> AppConfig:
    payload = {
        "app": {"data_dir": data_dir},
        "models": {
            "default_chat": "a",
            "default_extraction": "b",
            "default_code": "c",
            "default_embedding": "d",
        },
    }
    payload.update(overrides)
    return AppConfig.model_validate(payload)


class TestBuildAppContext:
    def test_wires_a_fully_populated_context(self, tmp_path):
        config = make_config(str(tmp_path / "data"))
        ctx = build_app_context(config, model_catalog_path=REPO_ROOT_CATALOG)

        assert ctx.config is config
        assert len(ctx.model_registry) == 10
        assert isinstance(ctx.runtime, FakeRuntime)
        assert ctx.data_dir.base_dir.is_dir()

    def test_defaults_to_a_fake_runtime_honest_skip(self, tmp_path):
        config = make_config(str(tmp_path / "data"))
        ctx = build_app_context(config, model_catalog_path=REPO_ROOT_CATALOG)
        assert isinstance(ctx.runtime, FakeRuntime)

    def test_accepts_an_injected_runtime(self, tmp_path):
        config = make_config(str(tmp_path / "data"))
        custom_runtime = FakeRuntime(default_response="custom")
        ctx = build_app_context(config, model_catalog_path=REPO_ROOT_CATALOG, runtime=custom_runtime)
        assert ctx.runtime is custom_runtime

    def test_admission_controller_reflects_the_config_limit(self, tmp_path):
        config = make_config(str(tmp_path / "data"), limits={"max_concurrent_requests": 2})
        ctx = build_app_context(config, model_catalog_path=REPO_ROOT_CATALOG)
        assert ctx.admission_controller.policy.max_concurrent_requests == 2

    async def test_the_wired_runtime_is_reachable_through_the_context(self, tmp_path):
        config = make_config(str(tmp_path / "data"))
        ctx = build_app_context(config, model_catalog_path=REPO_ROOT_CATALOG)
        response = await ctx.runtime.generate(LLMRequest(model="test", prompt="hello"))
        assert response.text

    def test_audit_log_is_backed_by_a_real_file_under_the_data_dir(self, tmp_path):
        config = make_config(str(tmp_path / "data"))
        ctx = build_app_context(config, model_catalog_path=REPO_ROOT_CATALOG)
        ctx.audit_log.record("trace-1", "lookup_order", {}, "success", "")
        assert ctx.data_dir.audit_db.exists()

from pathlib import Path

import pytest
from local_ai_core.deployment.model_registry import load_model_registry

from gw_router import TaskNotFoundError, UnknownModelInRouteError, load_routes, resolve_route

REPO_ROOT_CATALOG = "models/MODEL_CATALOG.md"
ROUTES_PATH = Path(__file__).resolve().parent.parent / "config" / "gateway_routes.yaml"


class TestLoadRoutes:
    def test_loads_the_real_committed_routes_config(self):
        registry = load_model_registry(REPO_ROOT_CATALOG)
        routes = load_routes(ROUTES_PATH, model_registry=registry)

        assert set(routes) == {"extraction", "code", "chat"}
        assert routes["chat"].primary_model == "llama3.1:8b-instruct"
        assert routes["chat"].fallback_model == "qwen2.5:1.5b-instruct"
        assert routes["chat"].max_context_tokens == 4096

    def test_raises_when_a_route_names_an_unknown_model(self, tmp_path):
        registry = load_model_registry(REPO_ROOT_CATALOG)
        bad_config = tmp_path / "routes.yaml"
        bad_config.write_text(
            "routes:\n  chat:\n    primary: does-not-exist\n    fallback: qwen2.5:1.5b-instruct\n"
            "    max_context_tokens: 4096\n"
        )
        with pytest.raises(UnknownModelInRouteError):
            load_routes(bad_config, model_registry=registry)

    def test_raises_when_the_fallback_model_is_unknown(self, tmp_path):
        registry = load_model_registry(REPO_ROOT_CATALOG)
        bad_config = tmp_path / "routes.yaml"
        bad_config.write_text(
            "routes:\n  chat:\n    primary: qwen2.5:1.5b-instruct\n    fallback: does-not-exist\n"
            "    max_context_tokens: 4096\n"
        )
        with pytest.raises(UnknownModelInRouteError):
            load_routes(bad_config, model_registry=registry)


class TestResolveRoute:
    def test_resolves_a_known_task(self):
        registry = load_model_registry(REPO_ROOT_CATALOG)
        routes = load_routes(ROUTES_PATH, model_registry=registry)
        route = resolve_route(routes, "code")
        assert route.primary_model == "qwen2.5-coder:7b"

    def test_raises_for_an_unknown_task(self):
        registry = load_model_registry(REPO_ROOT_CATALOG)
        routes = load_routes(ROUTES_PATH, model_registry=registry)
        with pytest.raises(TaskNotFoundError):
            resolve_route(routes, "does-not-exist")

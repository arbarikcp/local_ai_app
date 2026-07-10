from local_ai_core.deployment.model_registry import load_model_registry, parse_model_catalog

REPO_ROOT_CATALOG = "models/MODEL_CATALOG.md"


class TestParseModelCatalog:
    def test_parses_all_ten_real_entries(self):
        entries = parse_model_catalog(REPO_ROOT_CATALOG)
        assert len(entries) == 10

    def test_every_entry_has_a_real_model_id(self):
        entries = parse_model_catalog(REPO_ROOT_CATALOG)
        assert all(entry.model_id for entry in entries)
        assert "qwen2.5:1.5b-instruct" in {e.model_id for e in entries}

    def test_categories_match_the_real_catalog_composition(self):
        entries = parse_model_catalog(REPO_ROOT_CATALOG)
        categories = [e.category for e in entries]
        assert categories.count("chat") == 5
        assert categories.count("code") == 2
        assert categories.count("embedding") == 2
        assert categories.count("reranker") == 1

    def test_runtime_support_is_parsed_as_real_booleans(self):
        entries = parse_model_catalog(REPO_ROOT_CATALOG)
        first = next(e for e in entries if e.model_id == "qwen2.5:1.5b-instruct")
        assert first.runtime.ollama is True
        assert first.runtime.mlx is True


class TestModelRegistry:
    def test_get_returns_the_matching_entry(self):
        registry = load_model_registry(REPO_ROOT_CATALOG)
        entry = registry.get("qwen2.5:1.5b-instruct")
        assert entry is not None
        assert entry.family == "qwen2.5"

    def test_get_returns_none_for_an_unknown_model_id(self):
        registry = load_model_registry(REPO_ROOT_CATALOG)
        assert registry.get("does-not-exist") is None

    def test_by_category_filters_correctly(self):
        registry = load_model_registry(REPO_ROOT_CATALOG)
        code_models = registry.by_category("code")
        assert len(code_models) == 2
        assert all(e.category == "code" for e in code_models)

    def test_categories_returns_every_real_category(self):
        registry = load_model_registry(REPO_ROOT_CATALOG)
        assert registry.categories() == {"chat", "code", "embedding", "reranker"}

    def test_len_matches_the_real_entry_count(self):
        registry = load_model_registry(REPO_ROOT_CATALOG)
        assert len(registry) == 10

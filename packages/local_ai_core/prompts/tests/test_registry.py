import pytest

from local_ai_core.prompts.registry import PromptNotFoundError, PromptRegistry
from local_ai_core.prompts.template import PromptTemplate


def _template(prompt_id="extraction", version="v1") -> PromptTemplate:
    return PromptTemplate(prompt_id=prompt_id, version=version, role="role", task="task")


class TestRegisterAndGet:
    def test_register_then_get_returns_the_same_template(self):
        registry = PromptRegistry()
        template = _template()
        registry.register(template)
        assert registry.get("extraction", "v1") is template

    def test_get_without_version_returns_latest_registered(self):
        registry = PromptRegistry()
        registry.register(_template(version="v1"))
        v2 = _template(version="v2")
        registry.register(v2)
        assert registry.get("extraction") is v2

    def test_get_unknown_prompt_id_raises_prompt_not_found(self):
        registry = PromptRegistry()
        with pytest.raises(PromptNotFoundError):
            registry.get("does-not-exist")

    def test_get_unknown_version_raises_prompt_not_found(self):
        registry = PromptRegistry()
        registry.register(_template(version="v1"))
        with pytest.raises(PromptNotFoundError):
            registry.get("extraction", "v99")

    def test_reregistering_the_same_prompt_id_and_version_is_rejected(self):
        registry = PromptRegistry()
        registry.register(_template(version="v1"))
        with pytest.raises(ValueError, match="immutable"):
            registry.register(_template(version="v1"))

    def test_different_prompt_ids_are_independent(self):
        registry = PromptRegistry()
        registry.register(_template(prompt_id="extraction", version="v1"))
        registry.register(_template(prompt_id="classification", version="v1"))
        assert registry.get("extraction", "v1").prompt_id == "extraction"
        assert registry.get("classification", "v1").prompt_id == "classification"


class TestListVersions:
    def test_lists_all_registered_versions_sorted(self):
        registry = PromptRegistry()
        registry.register(_template(version="v2"))
        registry.register(_template(version="v1"))
        assert registry.list_versions("extraction") == ["v1", "v2"]

    def test_unknown_prompt_id_raises(self):
        registry = PromptRegistry()
        with pytest.raises(PromptNotFoundError):
            registry.list_versions("does-not-exist")


class TestListPromptIds:
    def test_lists_all_registered_prompt_ids(self):
        registry = PromptRegistry()
        registry.register(_template(prompt_id="extraction"))
        registry.register(_template(prompt_id="classification"))
        assert registry.list_prompt_ids() == ["classification", "extraction"]

    def test_empty_registry_returns_empty_list(self):
        assert PromptRegistry().list_prompt_ids() == []


class TestLatestVersion:
    def test_returns_the_most_recently_registered_version(self):
        registry = PromptRegistry()
        registry.register(_template(version="v1"))
        registry.register(_template(version="v2"))
        assert registry.latest_version("extraction") == "v2"

    def test_unknown_prompt_id_raises(self):
        registry = PromptRegistry()
        with pytest.raises(PromptNotFoundError):
            registry.latest_version("does-not-exist")


def test_prompt_not_found_error_message_distinguishes_missing_id_vs_version():
    id_error = PromptNotFoundError("extraction")
    version_error = PromptNotFoundError("extraction", "v99")
    assert "extraction" in str(id_error)
    assert "v99" in str(version_error)

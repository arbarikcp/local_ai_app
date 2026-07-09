import pytest

from local_ai_agents.tools.mcp_prompts import PromptArgumentError, PromptNotFoundError, PromptRegistry


class TestRegisterAndList:
    def test_lists_every_registered_prompt(self):
        registry = PromptRegistry()
        registry.register("greeting", "Hello, {name}!", "greets someone", argument_names=["name"])
        descriptors = registry.list()
        assert len(descriptors) == 1
        assert descriptors[0].name == "greeting"
        assert descriptors[0].argument_names == ["name"]

    def test_empty_registry_lists_nothing(self):
        assert PromptRegistry().list() == []


class TestGet:
    def test_renders_the_template_with_arguments(self):
        registry = PromptRegistry()
        registry.register("greeting", "Hello, {name}!", "greets someone", argument_names=["name"])
        assert registry.get("greeting", {"name": "Ada"}) == "Hello, Ada!"

    def test_a_prompt_with_no_arguments_needs_none_supplied(self):
        registry = PromptRegistry()
        registry.register("static", "This never changes.", "static text")
        assert registry.get("static") == "This never changes."

    def test_raises_for_an_unregistered_prompt(self):
        registry = PromptRegistry()
        with pytest.raises(PromptNotFoundError):
            registry.get("missing")

    def test_raises_when_a_required_argument_is_missing(self):
        registry = PromptRegistry()
        registry.register("greeting", "Hello, {name}!", "greets someone", argument_names=["name"])
        with pytest.raises(PromptArgumentError):
            registry.get("greeting", {})

    def test_the_rag_prompt_exemplar_renders_with_context_and_question(self):
        registry = PromptRegistry()
        registry.register(
            "rag_answer",
            "Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:",
            "Module 11's minimal RAG prompt, exposed as a reusable MCP-shaped prompt",
            argument_names=["context", "question"],
        )
        rendered = registry.get("rag_answer", {"context": "The sky is blue.", "question": "What color is the sky?"})
        assert "The sky is blue." in rendered
        assert "What color is the sky?" in rendered

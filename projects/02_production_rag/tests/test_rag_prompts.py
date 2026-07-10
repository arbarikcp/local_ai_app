from rag_prompts import RAG_PROMPT_VERSION, current_prompt_template, prompt_metadata


class TestPromptMetadata:
    def test_includes_the_real_version_string(self):
        metadata = prompt_metadata()
        assert metadata["version"] == RAG_PROMPT_VERSION

    def test_names_the_real_source_module(self):
        metadata = prompt_metadata()
        assert "citation_packer" in metadata["source"]


class TestCurrentPromptTemplate:
    def test_returns_the_real_curriculum_prompt(self):
        template = current_prompt_template()
        assert "Answer only using the provided context" in template
        assert "{context}" in template
        assert "{question}" in template

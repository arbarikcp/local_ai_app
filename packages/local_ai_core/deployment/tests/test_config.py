import pytest
from pydantic import ValidationError

from local_ai_core.deployment.config import AppConfig, load_config

REPO_ROOT_CONFIG = "config/app.example.yaml"

MINIMAL_YAML = """
models:
  default_chat: llama3.2:3b
  default_extraction: gemma3:4b
  default_code: qwen2.5-coder:7b
  default_embedding: nomic-embed-text
"""


class TestLoadConfig:
    def test_loads_the_real_committed_example_config(self):
        config = load_config(REPO_ROOT_CONFIG)
        assert config.app.data_dir == "~/.local-llm-ai"
        assert config.app.log_level == "INFO"
        assert config.models.default_chat == "llama3.2:3b"
        assert config.limits.max_concurrent_requests == 1
        assert config.security.allow_shell is False
        assert config.security.allow_file_write == "approval_required"

    def test_minimal_config_fills_in_real_defaults(self, tmp_path):
        path = tmp_path / "minimal.yaml"
        path.write_text(MINIMAL_YAML)
        config = load_config(path)
        assert config.app.data_dir == "~/.local-llm-ai"
        assert config.limits.max_prompt_tokens == 6000
        assert config.security.redact_pii_in_logs is True


class TestValidation:
    def test_missing_required_models_section_raises(self):
        with pytest.raises(ValidationError):
            AppConfig.model_validate({})

    def test_invalid_allow_file_write_value_raises(self):
        with pytest.raises(ValidationError):
            AppConfig.model_validate(
                {
                    "models": {
                        "default_chat": "a",
                        "default_extraction": "b",
                        "default_code": "c",
                        "default_embedding": "d",
                    },
                    "security": {"allow_file_write": "sometimes"},
                }
            )

    def test_nonpositive_max_concurrent_requests_raises(self):
        with pytest.raises(ValidationError):
            AppConfig.model_validate(
                {
                    "models": {
                        "default_chat": "a",
                        "default_extraction": "b",
                        "default_code": "c",
                        "default_embedding": "d",
                    },
                    "limits": {"max_concurrent_requests": 0},
                }
            )

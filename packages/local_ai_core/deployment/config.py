"""AppConfig — real Pydantic validation of curriculum's exact config shape
(theory doc §7, "Config example"). Loaded from a real YAML file via
`load_config()` - the first config loader anywhere in this repo.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class AppSection(BaseModel):
    data_dir: str = "~/.local-llm-ai"
    log_level: str = "INFO"
    offline_mode: bool = False


class ModelsSection(BaseModel):
    default_chat: str
    default_extraction: str
    default_code: str
    default_embedding: str


class LimitsSection(BaseModel):
    max_prompt_tokens: int = Field(gt=0, default=6000)
    max_output_tokens: int = Field(gt=0, default=1024)
    request_timeout_seconds: float = Field(gt=0, default=60)
    max_concurrent_requests: int = Field(gt=0, default=1)


class SecuritySection(BaseModel):
    allow_shell: bool = False
    allow_file_write: Literal["never", "approval_required", "always"] = "approval_required"
    redact_pii_in_logs: bool = True


class AppConfig(BaseModel):
    app: AppSection = Field(default_factory=AppSection)
    models: ModelsSection
    limits: LimitsSection = Field(default_factory=LimitsSection)
    security: SecuritySection = Field(default_factory=SecuritySection)


def load_config(path: str | Path) -> AppConfig:
    with Path(path).open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return AppConfig.model_validate(raw)

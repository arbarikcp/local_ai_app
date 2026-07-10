"""Labs 3-4 - add a config file, add a model registry. Loads the real,
committed `config/app.example.yaml` (real Pydantic validation) and parses
the real, committed `models/MODEL_CATALOG.md` (real YAML-fence extraction)
- both genuine, already-existing files, not fixtures invented for this demo.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "packages"))

from pydantic import ValidationError  # noqa: E402

from local_ai_core.deployment.config import AppConfig, load_config  # noqa: E402
from local_ai_core.deployment.model_registry import load_model_registry  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = REPO_ROOT / "config" / "app.example.yaml"
CATALOG_PATH = REPO_ROOT / "models" / "MODEL_CATALOG.md"


def run_lab() -> dict:
    config = load_config(CONFIG_PATH)

    # A deliberately invalid config, to prove validation actually rejects
    # something rather than accepting anything handed to it.
    try:
        AppConfig.model_validate(
            {
                "models": {
                    "default_chat": "a",
                    "default_extraction": "b",
                    "default_code": "c",
                    "default_embedding": "d",
                },
                "security": {"allow_file_write": "whenever_i_feel_like_it"},
            }
        )
        validation_rejected_bad_input = False
    except ValidationError:
        validation_rejected_bad_input = True

    registry = load_model_registry(CATALOG_PATH)

    return {
        "default_chat_model": config.models.default_chat,
        "max_concurrent_requests": config.limits.max_concurrent_requests,
        "redact_pii_in_logs": config.security.redact_pii_in_logs,
        "validation_rejected_bad_input": validation_rejected_bad_input,
        "registry_size": len(registry),
        "categories": sorted(registry.categories()),
        "chat_models": [e.model_id for e in registry.by_category("chat")],
    }


def result_to_markdown(result: dict) -> str:
    return (
        "# Labs 3-4 - config file and model registry\n\n"
        f"- Default chat model (from config): {result['default_chat_model']}\n"
        f"- max_concurrent_requests: {result['max_concurrent_requests']}\n"
        f"- redact_pii_in_logs: {result['redact_pii_in_logs']}\n"
        f"- Validation correctly rejected a bad `allow_file_write` value: "
        f"{result['validation_rejected_bad_input']}\n"
        f"- Model registry: {result['registry_size']} entries across {result['categories']}\n"
        f"- Chat models: {result['chat_models']}\n"
    )


def main(argv: list[str] | None = None) -> int:
    result = run_lab()
    print(result_to_markdown(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

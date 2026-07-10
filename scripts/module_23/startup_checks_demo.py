"""Lab 5 - add startup checks. Runs `run_startup_checks()` for real against
a real, self-contained temporary data directory (never the user's actual
`~/.local-llm-ai`, so running this demo repeatedly has no side effects on
the real machine), then against a deliberately broken config to prove a
real failure is caught, not just the happy path.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "packages"))

from local_ai_core.deployment.config import AppConfig  # noqa: E402
from local_ai_core.deployment.health import run_startup_checks  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CATALOG_PATH = REPO_ROOT / "models" / "MODEL_CATALOG.md"


def make_config(data_dir: str) -> AppConfig:
    return AppConfig.model_validate(
        {
            "app": {"data_dir": data_dir},
            "models": {
                "default_chat": "llama3.2:3b",
                "default_extraction": "gemma3:4b",
                "default_code": "qwen2.5-coder:7b",
                "default_embedding": "nomic-embed-text",
            },
        }
    )


def run_lab() -> dict:
    with tempfile.TemporaryDirectory(prefix="module23-startup-checks-") as tmp_dir:
        healthy_config = make_config(str(Path(tmp_dir) / "data"))
        healthy_results = run_startup_checks(healthy_config, model_catalog_path=CATALOG_PATH)

        broken_config = make_config(str(Path(tmp_dir) / "data"))
        broken_catalog_path = Path(tmp_dir) / "does_not_exist.md"
        broken_results = run_startup_checks(broken_config, model_catalog_path=broken_catalog_path)

    return {
        "healthy_all_passed": all(r.passed for r in healthy_results),
        "healthy_results": [(r.name, r.passed, r.detail) for r in healthy_results],
        "broken_all_passed": all(r.passed for r in broken_results),
        "broken_results": [(r.name, r.passed, r.detail) for r in broken_results],
    }


def result_to_markdown(result: dict) -> str:
    lines = ["# Lab 5 - startup checks", "", "## Healthy configuration"]
    for name, passed, detail in result["healthy_results"]:
        status = "PASS" if passed else "FAIL"
        lines.append(f"- [{status}] {name}: {detail}")
    lines.append(f"\nAll passed: {result['healthy_all_passed']}\n")
    lines.append("## Broken configuration (missing model catalog)")
    for name, passed, detail in result["broken_results"]:
        status = "PASS" if passed else "FAIL"
        lines.append(f"- [{status}] {name}: {detail}")
    lines.append(f"\nAll passed: {result['broken_all_passed']}\n")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    result = run_lab()
    print(result_to_markdown(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

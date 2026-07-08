"""Lab 6 — model metadata probes via Ollama's native /api/show endpoint.

The parser (``parse_show_response``) is pure and tested against a fixture
shaped like real Ollama /api/show output; the network call
(``get_model_metadata``) is a thin wrapper, honest-skip on this machine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx

DEFAULT_BASE_URL = "http://localhost:11434"


@dataclass(frozen=True)
class ModelMetadata:
    model: str
    family: str | None
    parameter_size: str | None
    quantization_level: str | None
    format: str | None
    template: str | None
    context_length: int | None
    raw: dict[str, Any] = field(default_factory=dict)


def parse_show_response(model: str, data: dict[str, Any]) -> ModelMetadata:
    """Parse the JSON body of a POST /api/show response.

    Real Ollama responses nest architecture details under "details" and
    (on newer versions) additional fields under "model_info" keyed like
    "<family>.context_length" - both are handled, since which one is
    present has varied across Ollama versions (theory doc §11: do not
    assume metadata shape is stable across runtime versions).
    """
    details = data.get("details", {}) or {}
    model_info = data.get("model_info", {}) or {}

    context_length = None
    for key, value in model_info.items():
        if key.endswith(".context_length"):
            context_length = value
            break

    return ModelMetadata(
        model=model,
        family=details.get("family"),
        parameter_size=details.get("parameter_size"),
        quantization_level=details.get("quantization_level"),
        format=details.get("format"),
        template=data.get("template"),
        context_length=context_length,
        raw=data,
    )


def get_model_metadata(model: str, *, base_url: str = DEFAULT_BASE_URL, timeout: float = 10.0) -> ModelMetadata:
    resp = httpx.post(f"{base_url}/api/show", json={"name": model}, timeout=timeout)
    resp.raise_for_status()
    return parse_show_response(model, resp.json())


def metadata_to_markdown(metadata: ModelMetadata) -> str:
    return (
        f"| Field | Value |\n|---|---|\n"
        f"| model | {metadata.model} |\n"
        f"| family | {metadata.family or 'n/a'} |\n"
        f"| parameter_size | {metadata.parameter_size or 'n/a'} |\n"
        f"| quantization_level | {metadata.quantization_level or 'n/a'} |\n"
        f"| format | {metadata.format or 'n/a'} |\n"
        f"| context_length | {metadata.context_length or 'n/a'} |\n"
    )


def main() -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    try:
        metadata = get_model_metadata(args.model)
    except httpx.HTTPError as exc:
        print(
            f"SKIPPED: could not reach Ollama at {DEFAULT_BASE_URL} to probe metadata for "
            f"'{args.model}': {exc}",
            file=sys.stderr,
        )
        return 1

    print(f"# Model metadata — {args.model}\n")
    print(metadata_to_markdown(metadata))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

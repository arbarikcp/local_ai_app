"""Lab 7 — runtime feature comparison matrix.

Turns "runtimes differ" (theory doc §11) from received wisdom into an
explicit, versioned table. Every entry is currently ``verified=False``
(populated from public documentation, matching this course's own
MODEL_CATALOG.md pattern of "documented, not measured" — see Module 3) and
must be flipped to ``verified=True`` with real behavioral notes once
actually exercised on a resourced Mac. Runtime feature support changes
across versions — re-verify before trusting an entry, the same rule as the
model catalog.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SupportLevel = Literal["yes", "partial", "no", "n/a", "unknown"]

CAVEAT = (
    "Every row below is documented from public runtime documentation as of this course's "
    "authoring, NOT measured on a live runtime (see the machine constraint in PROGRESS.md). "
    "Runtime feature support changes across versions - re-verify before depending on any "
    "'yes' here, exactly as models/MODEL_CATALOG.md requires for model claims."
)


@dataclass(frozen=True)
class FeatureSupport:
    level: SupportLevel
    verified: bool
    notes: str = ""


@dataclass(frozen=True)
class RuntimeFeatures:
    runtime: str
    structured_output: FeatureSupport
    grammar: FeatureSupport
    token_counting_endpoint: FeatureSupport
    streaming: FeatureSupport
    cancellation: FeatureSupport
    usage_reporting: FeatureSupport


KNOWN_FEATURE_MATRIX: dict[str, RuntimeFeatures] = {
    "ollama": RuntimeFeatures(
        runtime="ollama",
        structured_output=FeatureSupport(
            "yes", False, "`format` request field accepts \"json\" or a JSON schema"
        ),
        grammar=FeatureSupport(
            "no", False, "no user-facing GBNF grammar parameter; structured output goes through `format` instead"
        ),
        token_counting_endpoint=FeatureSupport(
            "partial", False, "no dedicated pre-flight /tokenize endpoint historically; prompt_eval_count is reported post-hoc in the generate response"
        ),
        streaming=FeatureSupport("yes", False, "NDJSON stream via /api/generate and /api/chat"),
        cancellation=FeatureSupport("yes", False, "closing the client connection is commonly reported to stop generation server-side"),
        usage_reporting=FeatureSupport("yes", False, "prompt_eval_count/eval_count/durations in the final response object"),
    ),
    "llama_cpp_native": RuntimeFeatures(
        runtime="llama.cpp (llama-server)",
        structured_output=FeatureSupport("yes", False, "JSON-schema-to-grammar conversion supported"),
        grammar=FeatureSupport("yes", False, "native GBNF grammar support - the runtime's flagship structured-output feature"),
        token_counting_endpoint=FeatureSupport("yes", False, "/tokenize and /detokenize endpoints"),
        streaming=FeatureSupport("yes", False, "SSE streaming via /completion and OpenAI-compatible /v1/chat/completions"),
        cancellation=FeatureSupport("yes", False, "closing the connection stops the associated generation slot"),
        usage_reporting=FeatureSupport("yes", False, "usage fields in OpenAI-compatible responses plus native timing fields"),
    ),
    "llama_cpp_python_server": RuntimeFeatures(
        runtime="llama-cpp-python[server]",
        structured_output=FeatureSupport("yes", False, "supports response_format / JSON-schema-constrained generation"),
        grammar=FeatureSupport("yes", False, "exposes llama.cpp's GBNF grammar support through its own API"),
        token_counting_endpoint=FeatureSupport("yes", False, "exposes a tokenize endpoint"),
        streaming=FeatureSupport("yes", False, "OpenAI-compatible SSE streaming"),
        cancellation=FeatureSupport("partial", False, "depends on server version/config; less consistently documented than native llama.cpp"),
        usage_reporting=FeatureSupport("yes", False, "OpenAI-compatible usage block"),
    ),
    "mlx_lm": RuntimeFeatures(
        runtime="MLX / mlx-lm",
        structured_output=FeatureSupport("no", False, "no built-in constrained decoding; would need an external library (e.g. outlines) layered on top"),
        grammar=FeatureSupport("no", False, "no built-in grammar support"),
        token_counting_endpoint=FeatureSupport("partial", False, "tokenizer accessible as a library call, not a server endpoint - mlx-lm has no built-in server"),
        streaming=FeatureSupport("yes", False, "mlx_lm.stream_generate yields tokens incrementally"),
        cancellation=FeatureSupport("n/a", False, "no built-in server process to cancel a connection against; in-process calls are cancelled by the caller simply stopping iteration"),
        usage_reporting=FeatureSupport("partial", False, "recent mlx-lm versions return generation stats (tokens, timing) from the library call, not a served 'usage' object"),
    ),
}

_FEATURE_COLUMNS = [
    ("structured_output", "Structured output"),
    ("grammar", "Grammar"),
    ("token_counting_endpoint", "Token counting"),
    ("streaming", "Streaming"),
    ("cancellation", "Cancellation"),
    ("usage_reporting", "Usage reporting"),
]


def _cell(support: FeatureSupport) -> str:
    marker = "measured" if support.verified else "documented"
    return f"{support.level} ({marker})"


def matrix_to_markdown(matrix: dict[str, RuntimeFeatures]) -> str:
    header_cols = ["Runtime"] + [label for _, label in _FEATURE_COLUMNS]
    header = "| " + " | ".join(header_cols) + " |\n"
    header += "|" + "|".join(["---"] * len(header_cols)) + "|\n"
    lines = [header.rstrip("\n")]
    for features in matrix.values():
        row_cells = [features.runtime]
        for attr, _label in _FEATURE_COLUMNS:
            row_cells.append(_cell(getattr(features, attr)))
        lines.append("| " + " | ".join(row_cells) + " |")
    return "\n".join(lines)


def notes_appendix(matrix: dict[str, RuntimeFeatures]) -> str:
    """Render every feature's notes field, since the summary table cells are terse."""
    lines = [f"> {CAVEAT}\n"]
    for features in matrix.values():
        lines.append(f"### {features.runtime}\n")
        for attr, label in _FEATURE_COLUMNS:
            support: FeatureSupport = getattr(features, attr)
            if support.notes:
                lines.append(f"- **{label}**: {support.notes}")
        lines.append("")
    return "\n".join(lines)


def unverified_entries(matrix: dict[str, RuntimeFeatures]) -> list[tuple[str, str]]:
    """List of (runtime, feature_label) pairs still pending real measurement."""
    pending = []
    for features in matrix.values():
        for attr, label in _FEATURE_COLUMNS:
            support: FeatureSupport = getattr(features, attr)
            if not support.verified:
                pending.append((features.runtime, label))
    return pending


def main() -> int:
    print("# Runtime feature matrix\n")
    print(matrix_to_markdown(KNOWN_FEATURE_MATRIX))
    print()
    print(notes_appendix(KNOWN_FEATURE_MATRIX))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

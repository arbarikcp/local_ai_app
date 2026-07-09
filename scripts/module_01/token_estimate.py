"""Pre-flight token estimation for Module 1 labs.

This module exists to make a single point stick: **never budget a local-model
prompt with `tiktoken` or a generic word-count heuristic** (see
docs/modules/01_local_llm_systems_thinking.md, section 5). It provides:

- ``estimate_tokens_heuristic``: an explicitly-labeled rough estimate, used
  only for building stress-test prompts of an approximate target length
  (Lab 1.2), never for authoritative budgeting.
- ``HFTokenizerCounter``: an optional exact counter backed by a real
  model tokenizer via ``transformers``, used when a tokenizer is available
  locally. This is the "count on the rendered prompt with the model's own
  tokenizer" path the theory doc requires.

Authoritative post-hoc counts should come from the runtime itself
(``GenerationObservation.prompt_eval_count`` / ``eval_count`` in
``ollama_probe.py``), not from anything in this file.
"""

from __future__ import annotations

from dataclasses import dataclass


# Empirically, English prose across common BPE-style tokenizers (Llama/Qwen/
# Gemma/Mistral families) averages roughly 1.3 tokens per whitespace-split
# word. This is a labeled approximation for building test prompts of a
# target length, not a budgeting tool.
_HEURISTIC_TOKENS_PER_WORD = 1.3


def estimate_tokens_heuristic(text: str) -> int:
    """Rough token estimate for building stress-test inputs.

    Deliberately not called ``count_tokens`` — the name makes it hard to
    mistake this for an authoritative count.
    """
    words = text.split()
    return max(1, round(len(words) * _HEURISTIC_TOKENS_PER_WORD))


def words_for_target_tokens(target_tokens: int) -> int:
    """Inverse of the heuristic: how many words to generate a prompt of
    roughly ``target_tokens`` tokens.
    """
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")
    return max(1, round(target_tokens / _HEURISTIC_TOKENS_PER_WORD))


@dataclass(frozen=True)
class TokenizerUnavailable(Exception):
    """Raised when an exact tokenizer backend cannot be loaded."""

    model_id: str

    def __str__(self) -> str:  # pragma: no cover - trivial
        return (
            f"Could not load tokenizer for '{self.model_id}'. Install `transformers` "
            "and ensure the model id is a valid Hugging Face repo, or fall back to "
            "runtime-reported counts."
        )


class HFTokenizerCounter:
    """Exact token counting using a Hugging Face tokenizer.

    Loaded lazily so that importing this module never requires
    ``transformers`` to be installed — only labs that explicitly need exact
    pre-flight counts pay that cost.

    Enabling this for real:
        1. In pyproject.toml, uncomment ``"transformers>=4.46"``, then run ``uv sync``.
        2. Pick a model id whose tokenizer is public on Hugging Face, e.g.
           ``"Qwen/Qwen2.5-1.5B-Instruct"`` — no separate download step,
           ``AutoTokenizer.from_pretrained(model_id)`` downloads and caches it on
           first use.
        3. Construct with no changes: ``HFTokenizerCounter(model_id)`` — the
           ``transformers`` import already lives inside ``_ensure_loaded()``; it
           only raises ``TokenizerUnavailable`` when the package isn't installed.
    """

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self._tokenizer = None

    def _ensure_loaded(self) -> None:
        if self._tokenizer is not None:
            return
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise TokenizerUnavailable(self.model_id) from exc
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        except Exception as exc:  # noqa: BLE001 - surfaced via TokenizerUnavailable
            raise TokenizerUnavailable(self.model_id) from exc

    def count(self, rendered_prompt: str) -> int:
        """Count tokens in an already chat-template-rendered prompt string."""
        self._ensure_loaded()
        assert self._tokenizer is not None
        return len(self._tokenizer.encode(rendered_prompt))

    def count_chat(self, messages: list[dict[str, str]]) -> int:
        """Count tokens after applying the model's own chat template.

        This is the correct way to budget a multi-turn prompt: special
        tokens, role markers, and separators the model actually sees are all
        included, unlike counting the raw message strings.
        """
        self._ensure_loaded()
        assert self._tokenizer is not None
        rendered = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return len(self._tokenizer.encode(rendered))

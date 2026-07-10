"""Code-specific prompt templates (ARCHITECTURE.md; PROPOSAL.md's survey
gap: "explain a function/class" and "suggest refactoring" have no prompt
templates anywhere in the repo - only `patch_tools.py`'s
`PATCH_PROMPT_TEMPLATE` and a script-inline test-generation template
exist). Patch proposal itself is NOT duplicated here - `patch_tools.py`'s
real template is reused unchanged by `eng_tools.make_propose_patch_tool()`.
"""

from __future__ import annotations

EXPLAIN_SYMBOL_PROMPT_TEMPLATE = """You are explaining a piece of code to a developer.

Symbol: {symbol_name}

Source:
{source_excerpt}

Explain what this symbol does, in plain language. Mention any edge cases or assumptions
visible in the code. Be concise."""


SUGGEST_REFACTOR_PROMPT_TEMPLATE = """You are suggesting a refactor for the following code.

Source:
{source_excerpt}

Suggest specific, concrete improvements (naming, structure, duplication, error handling).
Do not propose a patch - describe the changes in plain language only."""


GENERATE_TESTS_PROMPT_TEMPLATE = """You are writing a pytest test function for the following code.

Symbol: {symbol_name}

Source:
{source_excerpt}

Write one pytest test function covering the normal case and at least one edge case.
Respond with only the test function's source code, nothing else."""


def build_explain_symbol_prompt(symbol_name: str, source_excerpt: str) -> str:
    return EXPLAIN_SYMBOL_PROMPT_TEMPLATE.format(symbol_name=symbol_name, source_excerpt=source_excerpt)


def build_suggest_refactor_prompt(source_excerpt: str) -> str:
    return SUGGEST_REFACTOR_PROMPT_TEMPLATE.format(source_excerpt=source_excerpt)


def build_generate_tests_prompt(symbol_name: str, source_excerpt: str) -> str:
    return GENERATE_TESTS_PROMPT_TEMPLATE.format(symbol_name=symbol_name, source_excerpt=source_excerpt)

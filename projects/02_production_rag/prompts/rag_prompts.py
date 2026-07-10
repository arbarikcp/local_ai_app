"""RAG prompt version tracking (ARCHITECTURE.md, curriculum's own trace
field "prompt template version" - Module 21 theory doc §3). The actual
prompt text is Module 11's `RAG_PROMPT_TEMPLATE`/`build_rag_prompt()`
(curriculum.md §21's minimal RAG prompt, verbatim) used internally by the
reused `ProductionRagPipeline` - this module doesn't rebuild or wrap prompt
construction, only names the version so a stored query log entry could
record which prompt template version answered a given question, the same
discipline Project 1's `extraction_prompts.py` applied to extraction.
"""

from __future__ import annotations

from local_ai_rag.context_packers.citation_packer import RAG_PROMPT_TEMPLATE

RAG_PROMPT_VERSION = "v1"


def prompt_metadata() -> dict[str, str]:
    return {
        "version": RAG_PROMPT_VERSION,
        "source": "local_ai_rag.context_packers.citation_packer.RAG_PROMPT_TEMPLATE",
    }


def current_prompt_template() -> str:
    return RAG_PROMPT_TEMPLATE

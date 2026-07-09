"""Synthetic question generation (theory doc §2) - one LLM call per
document asking for N candidate questions, parsed into a list. Mechanically
real, `FakeRuntime`-backed here; a real model's question *quality* is
deferred to the resourced Mac, same discipline as Module 12's
`query_expansion.py`.
"""

from __future__ import annotations

from local_ai_core.runtimes.base import LLMRuntime
from local_ai_core.runtimes.types import LLMRequest

SYNTHETIC_QUESTION_PROMPT_TEMPLATE = """Read the following document and write {n} distinct \
questions that can be answered using only the information in the document. Reply with \
exactly {n} lines, one question per line, nothing else.

Document:
{document_text}

Questions:"""


async def generate_questions_from_document(
    document_text: str, runtime: LLMRuntime, model: str, n: int = 3
) -> list[str]:
    request = LLMRequest(
        model=model, prompt=SYNTHETIC_QUESTION_PROMPT_TEMPLATE.format(document_text=document_text, n=n)
    )
    response = await runtime.generate(request)
    questions = [line.strip() for line in response.text.splitlines() if line.strip()]
    return questions[:n]

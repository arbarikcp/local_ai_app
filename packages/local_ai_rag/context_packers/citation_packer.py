"""Prompt assembly and citation extraction (theory doc "Prompt assembly",
"Basic citations"). `build_rag_prompt()` implements curriculum.md §21's
minimal RAG prompt verbatim - small local models are sensitive to prompt
wording (Module 7), so this isn't paraphrased.
"""

from __future__ import annotations

import re

from local_ai_rag.embeddings.embedder import SearchResult

RAG_PROMPT_TEMPLATE = """You are a question answering assistant.
Answer only using the provided context.
If the answer is not present in the context, say: "I don't know based on the provided documents."

Context:
{context}

Question:
{question}

Answer:"""

_CITATION_RE = re.compile(r"\[([A-Za-z0-9_.-]+::\d+)\]")


def build_context(results: list[SearchResult]) -> str:
    """Each chunk is tagged with its citation key up front, `[chunk_id]`,
    so the model sees the exact string it should cite - not a separate
    "sources" list disconnected from the text it describes. `SearchResult
    .doc_id` holds the chunk_id here, since `NaiveRagPipeline` indexes
    chunks (not whole documents) keyed by `document_chunker.py`'s
    `chunk_id` (`f"{doc_id}::{chunk_index}"`).
    """
    return "\n\n".join(f"[{r.doc_id}] {r.text}" for r in results)


def build_rag_prompt(question: str, results: list[SearchResult]) -> str:
    return RAG_PROMPT_TEMPLATE.format(context=build_context(results), question=question)


def extract_citations(answer_text: str) -> list[str]:
    """Every unique `[doc_id::chunk_index]`-shaped marker in the answer,
    in first-seen order - the exact `chunk_id` format `document_chunker.py`
    produces, so a caller can cross-check citations against real chunk ids.
    """
    seen: list[str] = []
    for match in _CITATION_RE.finditer(answer_text):
        if match.group(1) not in seen:
            seen.append(match.group(1))
    return seen

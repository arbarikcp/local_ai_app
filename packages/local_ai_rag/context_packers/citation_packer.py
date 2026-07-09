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

_CITATION_RE = re.compile(r"\[([A-Za-z0-9_.:-]+::\d+)\]")


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
    """Every unique `[...::chunk_index]`-shaped marker in the answer, in
    first-seen order - the exact `chunk_id` format `document_chunker.py`
    produces (`doc_id::chunk_index`, or Module 18's `pdf_stem::pageN::
    chunk_index`), so a caller can cross-check citations against real
    chunk ids regardless of how many "::"-separated segments the doc_id
    itself has - only the final segment is required to be the numeric
    chunk index.
    """
    seen: list[str] = []
    for match in _CITATION_RE.finditer(answer_text):
        if match.group(1) not in seen:
            seen.append(match.group(1))
    return seen


def summarize_source_citations(chunk_citations: list[str]) -> list[str]:
    """Aggregates chunk-level citations (`doc_id::chunk_index`) up to
    unique, deduplicated document-level citations (`doc_id`), in
    first-seen order - Lab 5's "source-level citations": an end user
    reading "see password_reset" doesn't need to know it came from chunk 2
    specifically, and two citations from the same document shouldn't read
    as two separate sources.

    Strips only the *trailing* `::chunk_index` segment (`rsplit`, not
    `split`), not everything after the first "::" - a Module 18 PDF-page
    doc_id like `sample_invoice::page1::0` has two "::"-separated
    segments before the chunk index, and the source should stay
    `sample_invoice::page1` (the real page), not collapse to just
    `sample_invoice` and lose which page the citation came from.
    """
    seen: list[str] = []
    for chunk_id in chunk_citations:
        doc_id = chunk_id.rsplit("::", 1)[0]
        if doc_id not in seen:
            seen.append(doc_id)
    return seen

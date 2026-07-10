"""Prompt assembly and page-citation extraction for document Q&A
(ARCHITECTURE.md "Q&A with page citations"). `build_doc_qa_prompt()`
mirrors Module 11's `build_rag_prompt()` template shape (curriculum.md
§21's minimal RAG prompt), each page tagged `[page_id]` the same way
`citation_packer.build_context()` tags each chunk `[chunk_id]`.

A real finding, not an assumption: `citation_packer.extract_citations()`'s
regex requires a citation marker to end in `::<digits>` (a chunk index,
e.g. `doc::page1::0`) - it does NOT match a bare page id like
`multi_page_form::page2` (confirmed: `extract_citations("... [multi_page_
form::page2].")` returns `[]`). This project's page ids have no trailing
chunk index, so `extract_page_citations()` below is a new, page-id-shaped
regex, not a reuse of `citation_packer`'s. `citations_are_grounded()`
(Project 2, reused unchanged) is still fully reusable downstream, since it
only does set-membership - id-format-agnostic, confirmed by survey.
"""

from __future__ import annotations

import re

DOC_QA_PROMPT_TEMPLATE = """You are a document analysis assistant.
Answer only using the provided page context.
If the answer is not present in the context, say: "I don't know based on the provided document."
Cite every page you use with its exact page id in square brackets, e.g. [{example_page_id}].

Context:
{context}

Question:
{question}

Answer:"""

EXAMPLE_PAGE_ID = "multi_page_form::page1"

_PAGE_CITATION_RE = re.compile(r"\[([A-Za-z0-9_.-]+::page\d+)\]")


def build_doc_context(pages: list) -> str:
    """Each analyzed page is tagged `[page_id]` up front so the model sees
    the exact string it should cite. Quarantined pages have no
    `extracted_text` and are never passed in here in the first place
    (`doc_qa.answer_question()`'s job, not this function's).
    """
    return "\n\n".join(f"[{page.page_id}] {page.extracted_text}" for page in pages)


def build_doc_qa_prompt(question: str, pages: list) -> str:
    return DOC_QA_PROMPT_TEMPLATE.format(
        example_page_id=EXAMPLE_PAGE_ID, context=build_doc_context(pages), question=question
    )


def extract_page_citations(answer_text: str) -> list[str]:
    """Every unique `[doc_id::pageN]`-shaped marker in the answer, in
    first-seen order.
    """
    seen: list[str] = []
    for match in _PAGE_CITATION_RE.finditer(answer_text):
        if match.group(1) not in seen:
            seen.append(match.group(1))
    return seen

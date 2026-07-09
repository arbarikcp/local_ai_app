"""Lab 6 - add page/region citations: real PDF pages flow through Module
11's exact `NaiveRagPipeline` unchanged, and the resulting citations
reference real page numbers (`sample_invoice::page1`), not a new citation
mechanism. Runs for real except the final answer generation
(`FakeRuntime`).
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "packages"))

from local_ai_core.runtimes.fake import FakeRuntime  # noqa: E402
from local_ai_rag.context_packers.citation_packer import summarize_source_citations  # noqa: E402
from local_ai_rag.embeddings.fake import FakeEmbedder  # noqa: E402
from local_ai_rag.loaders.pdf_loader import load_pdf_directory  # noqa: E402
from local_ai_rag.pipeline import NaiveRagPipeline  # noqa: E402
from local_ai_rag.stores.numpy_store import NumpyVectorStore  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MULTIMODAL_DIR = REPO_ROOT / "datasets" / "multimodal"

QUESTION = "What is the invoice number?"
SCRIPTED_ANSWER = "The invoice number is INV-2026-0042 [sample_invoice::page1::0]."


async def run_lab() -> dict:
    embedder = FakeEmbedder(dimensions=32)
    store = NumpyVectorStore()
    runtime = FakeRuntime(default_response=SCRIPTED_ANSWER)
    pipeline = NaiveRagPipeline(embedder, store, runtime, model="fake-model", chunk_max_chars=500)

    documents = load_pdf_directory(MULTIMODAL_DIR)
    chunks = await pipeline.ingest_documents(documents)

    result = await pipeline.answer(QUESTION, k=3)

    return {
        "pdf_documents_loaded": [d.doc_id for d in documents],
        "chunks_ingested": [c.chunk_id for c in chunks],
        "question": QUESTION,
        "answer_text": result.answer_text,
        "chunk_level_citations": result.citations,
        "source_level_citations": summarize_source_citations(result.citations),
        "citations_are_grounded": result.citations_are_grounded,
    }


def result_to_markdown(result: dict) -> str:
    return (
        "# Lab 6 - page citations flowing through Module 11's RAG pipeline unchanged\n\n"
        f"- PDF-derived documents loaded: {result['pdf_documents_loaded']}\n"
        f"- Chunks ingested: {result['chunks_ingested']}\n"
        f"- Question: {result['question']}\n"
        f"- Answer: {result['answer_text']}\n"
        f"- Chunk-level citations: {result['chunk_level_citations']}\n"
        f"- Source-level (page) citations: {result['source_level_citations']}\n"
        f"- Citations grounded in retrieved chunks: {result['citations_are_grounded']}\n"
    )


def main(argv: list[str] | None = None) -> int:
    result = asyncio.run(run_lab())
    print(result_to_markdown(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

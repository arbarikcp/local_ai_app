"""Labs 1-2 — build naive RAG over the 20-file Nimbus handbook corpus and
add citations using chunk IDs.

Runs for real through retrieval and prompt assembly; answer generation
uses `FakeRuntime` with a scripted response that includes a real citation
marker, since a live LLM runtime isn't installed on this machine (see the
theory doc's machine note) - `citations_are_grounded` is exercised for
real regardless, since it only depends on comparing strings, not on what
generated them.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "packages"))

from local_ai_core.runtimes.fake import FakeRuntime  # noqa: E402
from local_ai_rag.embeddings.fake import FakeEmbedder  # noqa: E402
from local_ai_rag.pipeline import NaiveRagPipeline, RagAnswer  # noqa: E402
from local_ai_rag.stores.numpy_store import NumpyVectorStore  # noqa: E402

CORPUS_DIR = Path(__file__).resolve().parent.parent.parent / "datasets" / "rag_docs" / "nimbus_handbook"

DEMO_QUESTION = "How long does a password reset link stay valid?"
DEMO_RESPONSE = "The password reset link expires in 15 minutes [password_reset::0]."


def make_pipeline(chunk_max_chars: int = 500) -> NaiveRagPipeline:
    embedder = FakeEmbedder(dimensions=64)
    store = NumpyVectorStore()
    runtime = FakeRuntime(default_response=DEMO_RESPONSE)
    return NaiveRagPipeline(embedder, store, runtime, model="fake-model", chunk_max_chars=chunk_max_chars)


async def run_lab() -> dict:
    pipeline = make_pipeline()
    chunks = await pipeline.ingest_directory(CORPUS_DIR)

    result: RagAnswer = await pipeline.answer(DEMO_QUESTION, k=3)

    return {
        "documents_ingested": len({c.doc_id for c in chunks}),
        "chunks_ingested": len(chunks),
        "question": DEMO_QUESTION,
        "retrieved_chunk_ids": [r.doc_id for r in result.retrieved_chunks],
        "answer_text": result.answer_text,
        "citations": result.citations,
        "citations_are_grounded": result.citations_are_grounded,
    }


def result_to_markdown(result: dict) -> str:
    return (
        "# Labs 1-2 — naive RAG over the Nimbus handbook, with citations\n\n"
        f"- Documents ingested: {result['documents_ingested']}\n"
        f"- Chunks ingested: {result['chunks_ingested']}\n"
        f"- Question: {result['question']}\n"
        f"- Retrieved chunk ids: {result['retrieved_chunk_ids']}\n"
        f"- Answer: {result['answer_text']}\n"
        f"- Citations: {result['citations']}\n"
        f"- Citations grounded in retrieved chunks: {result['citations_are_grounded']}\n"
    )


def main(argv: list[str] | None = None) -> int:
    result = asyncio.run(run_lab())
    print(result_to_markdown(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

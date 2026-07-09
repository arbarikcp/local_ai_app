"""Lab 1 - parent-child retrieval over the Module 11 Nimbus handbook
corpus: index small child chunks, retrieve large parent chunks. Runs for
real - `FakeEmbedder`, no live model needed.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "packages"))

from local_ai_rag.chunkers.document_chunker import chunk_documents  # noqa: E402
from local_ai_rag.chunkers.parent_child_chunker import chunk_documents_parent_child  # noqa: E402
from local_ai_rag.embeddings.fake import FakeEmbedder  # noqa: E402
from local_ai_rag.loaders.markdown_loader import load_markdown_directory  # noqa: E402
from local_ai_rag.retrievers.parent_child_retriever import ParentChildRetriever  # noqa: E402
from local_ai_rag.stores.numpy_store import NumpyVectorStore  # noqa: E402

CORPUS_DIR = Path(__file__).resolve().parent.parent.parent / "datasets" / "rag_docs" / "nimbus_handbook"

QUESTION = "How long does a password reset link stay valid?"


async def build_parent_child_retriever(parent_max_chars: int = 800, child_max_chars: int = 150):
    documents = load_markdown_directory(CORPUS_DIR)
    embedder = FakeEmbedder(dimensions=64)
    store = NumpyVectorStore()
    index = chunk_documents_parent_child(documents, parent_max_chars=parent_max_chars, child_max_chars=child_max_chars)

    if index.children:
        vectors = await embedder.embed_documents([c.text for c in index.children])
        for child, vector in zip(index.children, vectors):
            await store.add(child.chunk_id, child.text, vector, metadata={"parent_id": child.parent_id})

    return ParentChildRetriever(embedder, store, index), documents, index


async def run_lab() -> dict:
    retriever, documents, index = await build_parent_child_retriever()
    parent_child_results = await retriever.retrieve(QUESTION, k=2)

    # For comparison: naive (flat) chunking over the same corpus, same chunk size as the parent.
    flat_chunks = chunk_documents(documents, max_chars=800)

    return {
        "documents": len(documents),
        "parent_chunks": len(index.parents),
        "child_chunks": len(index.children),
        "flat_chunks_at_parent_size": len(flat_chunks),
        "question": QUESTION,
        "top_parent_id": parent_child_results[0].parent_id if parent_child_results else None,
        "top_parent_text_length": len(parent_child_results[0].text) if parent_child_results else 0,
    }


def result_to_markdown(result: dict) -> str:
    return (
        "# Lab 1 - parent-child retrieval\n\n"
        f"- Documents: {result['documents']}\n"
        f"- Parent chunks: {result['parent_chunks']} (vs. {result['flat_chunks_at_parent_size']} flat chunks at the same size)\n"
        f"- Child chunks indexed: {result['child_chunks']}\n"
        f"- Question: {result['question']}\n"
        f"- Top parent id: {result['top_parent_id']}\n"
        f"- Top parent text length: {result['top_parent_text_length']} chars (retrieved via a much smaller child match)\n"
    )


def main(argv: list[str] | None = None) -> int:
    result = asyncio.run(run_lab())
    print(result_to_markdown(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

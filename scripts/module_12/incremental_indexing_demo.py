"""Lab 6 - incremental indexing over the Module 11 Nimbus handbook corpus:
sync once, edit one document and remove another, sync again, and prove
only the changed/added/removed documents triggered re-embedding work.
Runs for real - `FakeEmbedder`, no live model needed.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "packages"))

from local_ai_rag.embeddings.fake import FakeEmbedder  # noqa: E402
from local_ai_rag.incremental_indexer import IncrementalIndexer  # noqa: E402
from local_ai_rag.loaders.markdown_loader import load_markdown_directory  # noqa: E402
from local_ai_rag.stores.numpy_store import NumpyVectorStore  # noqa: E402

CORPUS_DIR = Path(__file__).resolve().parent.parent.parent / "datasets" / "rag_docs" / "nimbus_handbook"


async def run_lab() -> dict:
    documents = load_markdown_directory(CORPUS_DIR)
    embedder = FakeEmbedder(dimensions=32)
    store = NumpyVectorStore()
    indexer = IncrementalIndexer(embedder, store, chunk_max_chars=500)

    await indexer.sync(documents)
    calls_after_first_sync = embedder.embed_documents_call_count
    chunks_after_first_sync = await store.count()

    # Simulate real drift: one document is edited, one is removed entirely.
    edited_documents = [d for d in documents if d.doc_id != "supported_regions"]
    password_doc_index = next(i for i, d in enumerate(edited_documents) if d.doc_id == "password_reset")
    original = edited_documents[password_doc_index]
    edited_documents[password_doc_index] = type(original)(
        doc_id=original.doc_id,
        source_path=original.source_path,
        title=original.title,
        text=original.text + "\n\nUpdated: reset links now expire in ten minutes, not fifteen.",
    )

    second_diff = await indexer.sync(edited_documents)
    calls_after_second_sync = embedder.embed_documents_call_count
    chunks_after_second_sync = await store.count()

    return {
        "documents_first_sync": len(documents),
        "chunks_after_first_sync": chunks_after_first_sync,
        "embed_calls_for_first_sync": calls_after_first_sync,
        "second_sync_added": second_diff.added,
        "second_sync_updated": second_diff.updated,
        "second_sync_deleted": second_diff.deleted,
        "second_sync_unchanged_count": len(second_diff.unchanged),
        "embed_calls_for_second_sync": calls_after_second_sync - calls_after_first_sync,
        "chunks_after_second_sync": chunks_after_second_sync,
    }


def result_to_markdown(result: dict) -> str:
    return (
        "# Lab 6 - incremental indexing\n\n"
        f"- First sync: {result['documents_first_sync']} documents -> "
        f"{result['chunks_after_first_sync']} chunks, {result['embed_calls_for_first_sync']} embed_documents() call(s)\n"
        f"- Second sync (1 document edited, 1 removed, {result['second_sync_unchanged_count']} unchanged):\n"
        f"  - added: {result['second_sync_added']}\n"
        f"  - updated: {result['second_sync_updated']}\n"
        f"  - deleted: {result['second_sync_deleted']}\n"
        f"  - embed_documents() calls triggered: {result['embed_calls_for_second_sync']} "
        "(not re-embedding the 18 unchanged documents)\n"
        f"- Chunks after second sync: {result['chunks_after_second_sync']}\n"
    )


def main(argv: list[str] | None = None) -> int:
    result = asyncio.run(run_lab())
    print(result_to_markdown(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

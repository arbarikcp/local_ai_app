"""Labs 2-5 - metadata/ACL filtering, reranking, context packing, and
source-level citations, all wired through `ProductionRagPipeline` over the
Module 11 Nimbus handbook corpus. Runs for real - `FakeEmbedder` and
`HeuristicReranker`, `FakeRuntime` only for the final generation call.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "packages"))

from local_ai_core.runtimes.fake import FakeRuntime  # noqa: E402
from local_ai_rag.chunkers.document_chunker import chunk_documents  # noqa: E402
from local_ai_rag.context_packers.budget_packer import ContextBudget  # noqa: E402
from local_ai_rag.embeddings.fake import FakeEmbedder  # noqa: E402
from local_ai_rag.loaders.markdown_loader import load_markdown_directory  # noqa: E402
from local_ai_rag.production_pipeline import ProductionRagPipeline  # noqa: E402
from local_ai_rag.retrievers.acl import ACLPredicate, clearance_predicate  # noqa: E402
from local_ai_rag.stores.numpy_store import NumpyVectorStore  # noqa: E402

CORPUS_DIR = Path(__file__).resolve().parent.parent.parent / "datasets" / "rag_docs" / "nimbus_handbook"

# security_incident_response is tagged restricted, simulating an internal-only runbook
# mixed into an otherwise public handbook - a realistic ACL scenario.
RESTRICTED_DOC_IDS = {"security_incident_response"}

QUESTION = "How do I reset my password and what should I do if I suspect unauthorized access?"


async def build_pipeline(
    context_budget: ContextBudget | None = None, acl_predicate: ACLPredicate | None = None
) -> ProductionRagPipeline:
    documents = load_markdown_directory(CORPUS_DIR)
    embedder = FakeEmbedder(dimensions=64)
    store = NumpyVectorStore()
    chunks = chunk_documents(documents, max_chars=500)

    vectors = await embedder.embed_documents([c.text for c in chunks])
    for chunk, vector in zip(chunks, vectors):
        security_level = 5 if chunk.doc_id in RESTRICTED_DOC_IDS else 0
        await store.add(
            chunk.chunk_id, chunk.text, vector, metadata={"doc_id": chunk.doc_id, "security_level": security_level}
        )

    runtime = FakeRuntime(
        default_response=(
            "Reset your password from the sign-in page [password_reset::0]. "
            "If you suspect unauthorized access, reset your password immediately "
            "[security_incident_response::0]."
        )
    )
    return ProductionRagPipeline(
        embedder, store, runtime, model="fake-model", context_budget=context_budget, acl_predicate=acl_predicate
    )


async def run_lab() -> dict:
    low_clearance_pipeline = await build_pipeline(acl_predicate=clearance_predicate(user_clearance=0))
    low_clearance_result = await low_clearance_pipeline.answer(QUESTION, k=5)

    high_clearance_pipeline = await build_pipeline(acl_predicate=clearance_predicate(user_clearance=10))
    high_clearance_result = await high_clearance_pipeline.answer(QUESTION, k=5)

    tight_pipeline = await build_pipeline(
        context_budget=ContextBudget(
            max_context_tokens=60, reserved_for_system=0, reserved_for_question=0, reserved_for_answer=0
        )
    )
    tight_result = await tight_pipeline.answer(QUESTION, k=5)

    return {
        "question": QUESTION,
        "low_clearance_packed_doc_ids": sorted({c.doc_id.split("::")[0] for c in low_clearance_result.packed_chunks}),
        "high_clearance_packed_doc_ids": sorted({c.doc_id.split("::")[0] for c in high_clearance_result.packed_chunks}),
        "low_clearance_source_citations": low_clearance_result.source_citations,
        "high_clearance_source_citations": high_clearance_result.source_citations,
        "low_clearance_citations_grounded": low_clearance_result.citations_are_grounded,
        "high_clearance_citations_grounded": high_clearance_result.citations_are_grounded,
        "generous_budget_chunks_packed": high_clearance_result.trace.chunks_packed,
        "tight_budget_chunks_packed": tight_result.trace.chunks_packed,
    }


def result_to_markdown(result: dict) -> str:
    return (
        "# Labs 2-5 - ACL filtering, reranking, context packing, source-level citations\n\n"
        f"- Question: {result['question']}\n"
        f"- Low-clearance user sees documents from: {result['low_clearance_packed_doc_ids']}\n"
        f"- High-clearance user sees documents from: {result['high_clearance_packed_doc_ids']}\n"
        f"- Low-clearance source citations: {result['low_clearance_source_citations']} "
        f"(grounded: {result['low_clearance_citations_grounded']})\n"
        f"- High-clearance source citations: {result['high_clearance_source_citations']} "
        f"(grounded: {result['high_clearance_citations_grounded']})\n"
        f"- Chunks packed with a generous budget: {result['generous_budget_chunks_packed']}\n"
        f"- Chunks packed with a tight budget: {result['tight_budget_chunks_packed']}\n"
    )


def main(argv: list[str] | None = None) -> int:
    result = asyncio.run(run_lab())
    print(result_to_markdown(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

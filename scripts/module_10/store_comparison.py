"""Labs 1-4, 6 — store the same corpus in Chroma, LanceDB, and NumPy behind
one `VectorStore` interface, verify they agree, exercise metadata filters,
and demonstrate hybrid search.

Runs for real against all three backends - Chroma and LanceDB are vector
database libraries, not LLM runtimes, so they're installed on this machine
(unlike Module 9's real embedding models). Embeddings come from Module 9's
`FakeEmbedder`, a genuine bag-of-words hashing embedder, for the same
reason Module 9 used it: real, non-fabricated retrieval behavior without a
downloaded neural model.
"""

from __future__ import annotations

import asyncio
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "packages"))

from local_ai_rag.embeddings.fake import FakeEmbedder  # noqa: E402
from local_ai_rag.stores.chroma_store import ChromaVectorStore  # noqa: E402
from local_ai_rag.stores.hybrid import hybrid_search  # noqa: E402
from local_ai_rag.stores.lancedb_store import LanceDBVectorStore  # noqa: E402
from local_ai_rag.stores.numpy_store import NumpyVectorStore  # noqa: E402
from local_ai_rag.stores.vector_store import VectorStore  # noqa: E402

CORPUS: dict[str, tuple[str, dict[str, str]]] = {
    "doc_password": ("How to reset your password", {"category": "account"}),
    "doc_billing": ("Update your billing information and payment method", {"category": "billing"}),
    "doc_shipping": ("Track your shipment and delivery status", {"category": "shipping"}),
    "doc_password2": ("Forgot password recovery steps for your account", {"category": "account"}),
    "doc_order_code": ("Your order ACC88213 has shipped and is on its way", {"category": "shipping"}),
}


def make_stores(tmp_dir: str, dimensions: int) -> dict[str, VectorStore]:
    return {
        "numpy": NumpyVectorStore(),
        "chroma": ChromaVectorStore("module10-comparison", path=f"{tmp_dir}/chroma"),
        "lancedb": LanceDBVectorStore("module10-comparison", path=f"{tmp_dir}/lancedb", dimensions=dimensions),
    }


async def populate(stores: dict[str, VectorStore], embedder: FakeEmbedder) -> dict[str, list]:
    texts = [text for text, _metadata in CORPUS.values()]
    vectors = await embedder.embed_documents(texts)
    for store in stores.values():
        for (doc_id, (text, metadata)), vector in zip(CORPUS.items(), vectors):
            await store.add(doc_id, text, vector, metadata=metadata)
    return vectors


async def run_lab(tmp_dir: str, k: int = 3) -> dict:
    embedder = FakeEmbedder(dimensions=32)
    stores = make_stores(tmp_dir, embedder.dimensions)
    await populate(stores, embedder)

    query_text = "I forgot my password"
    query_vector = await embedder.embed_query(query_text)

    top_result_by_store = {}
    for name, store in stores.items():
        results = await store.search(query_vector, k=k)
        top_result_by_store[name] = results[0].doc_id if results else None

    filtered_by_store = {}
    for name, store in stores.items():
        results = await store.search(query_vector, k=k, metadata_filter={"category": "account"})
        filtered_by_store[name] = [r.doc_id for r in results]

    documents = {doc_id: text for doc_id, (text, _metadata) in CORPUS.items()}
    vector_only = await stores["numpy"].search(await embedder.embed_query("ACC88213"), k=1)
    hybrid_results = await hybrid_search(
        stores["numpy"], documents, query="ACC88213", query_embedding=await embedder.embed_query("ACC88213"), k=3
    )

    return {
        "top_result_by_store": top_result_by_store,
        "agree_across_backends": len(set(top_result_by_store.values())) == 1,
        "filtered_by_store": filtered_by_store,
        "vector_only_top_result_for_order_code_query": vector_only[0].doc_id if vector_only else None,
        "hybrid_result_ids": [r.doc_id for r in hybrid_results],
    }


def result_to_markdown(result: dict) -> str:
    lines = ["# Labs 1-4, 6 — store comparison, metadata filters, hybrid search\n"]
    lines.append("## Top result per backend (same corpus, same query)\n")
    for name, doc_id in result["top_result_by_store"].items():
        lines.append(f"- {name}: {doc_id}")
    lines.append(f"\nAll backends agree: {result['agree_across_backends']}\n")
    lines.append("## Metadata filter (category=account) per backend\n")
    for name, ids in result["filtered_by_store"].items():
        lines.append(f"- {name}: {ids}")
    lines.append("\n## Hybrid search recovers an exact-code match vector search alone misses\n")
    lines.append(f"- vector-only top result for query 'ACC88213': {result['vector_only_top_result_for_order_code_query']}")
    lines.append(f"- hybrid_search result ids: {result['hybrid_result_ids']}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    tmp_dir = tempfile.mkdtemp(prefix="module10-")
    try:
        result = asyncio.run(run_lab(tmp_dir))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    print(result_to_markdown(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

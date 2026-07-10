"""RagAppContext — the composition root for this project, extending (not
replacing) Module 23's `AppContext` with an embedder, a real persistent
vector store, and the metadata store (ARCHITECTURE.md "Deployment shape").
Same pattern Project 1's `extraction_service.py` established for Module
23's composition root.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from local_ai_core.deployment.app_context import AppContext, build_app_context
from local_ai_core.deployment.config import AppConfig
from local_ai_core.runtimes.base import LLMRuntime
from local_ai_rag.embeddings.embedder import Embedder
from local_ai_rag.embeddings.fake import FakeEmbedder
from local_ai_rag.stores.lancedb_store import LanceDBVectorStore
from local_ai_rag.stores.vector_store import VectorStore

from rag_metadata_store import RagMetadataStore


@dataclass
class RagAppContext:
    base: AppContext
    embedder: Embedder
    store: VectorStore
    metadata_store: RagMetadataStore


def build_rag_context(
    config: AppConfig,
    *,
    model_catalog_path: str | Path,
    runtime: LLMRuntime | None = None,
    embedder: Embedder | None = None,
) -> RagAppContext:
    """`embedder` defaults to `FakeEmbedder` - this repo's standing
    honest-skip default (real bag-of-words embedding, not a neural model);
    `SentenceTransformersEmbedder`/`OllamaEmbedder` are the documented
    "enable for real" path on the resourced Mac, no other code change
    needed since `RagAppContext` takes the embedder via dependency
    injection, same as `AppContext` already does for the runtime.
    """
    base = build_app_context(config, model_catalog_path=model_catalog_path, runtime=runtime)
    rag_dir = base.data_dir.base_dir / "rag"
    rag_dir.mkdir(parents=True, exist_ok=True)

    resolved_embedder = embedder or FakeEmbedder()
    store = LanceDBVectorStore(
        "chunks", path=str(rag_dir / "lancedb"), dimensions=resolved_embedder.dimensions
    )
    metadata_store = RagMetadataStore(rag_dir / "rag_metadata.db")

    return RagAppContext(base=base, embedder=resolved_embedder, store=store, metadata_store=metadata_store)

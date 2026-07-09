"""NaiveRagPipeline — the naive RAG architecture end to end (theory doc,
top-level diagram):

    Documents -> chunk -> embed chunks -> store vectors
    Query -> embed query -> top-k search -> build prompt -> local LLM answer

Every stage but the last runs for real on this machine. `answer()` calls
`runtime.generate()` against Module 6's `LLMRuntime` protocol via
dependency injection - `FakeRuntime` here, a real adapter unchanged on the
resourced Mac (see the theory doc's machine note).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from local_ai_core.runtimes.base import LLMRuntime
from local_ai_core.runtimes.types import LLMRequest
from local_ai_rag.chunkers.document_chunker import Chunk, chunk_documents
from local_ai_rag.context_packers.citation_packer import build_rag_prompt, extract_citations
from local_ai_rag.embeddings.embedder import Embedder, SearchResult
from local_ai_rag.loaders.markdown_loader import Document, load_markdown_directory
from local_ai_rag.stores.vector_store import VectorStore


@dataclass(frozen=True)
class RagAnswer:
    question: str
    answer_text: str
    retrieved_chunks: list[SearchResult]
    citations: list[str] = field(default_factory=list)

    @property
    def citations_are_grounded(self) -> bool:
        """False if the answer cites a chunk_id that wasn't actually
        retrieved - a detectable, measurable stand-in for "the model
        invented a citation" (theory doc's "Citations may be invented"
        gotcha), not just a documented risk.
        """
        retrieved_ids = {r.doc_id for r in self.retrieved_chunks}
        return all(citation in retrieved_ids for citation in self.citations)


class NaiveRagPipeline:
    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        runtime: LLMRuntime,
        *,
        model: str = "fake-model",
        chunk_max_chars: int = 500,
        chunk_overlap_chars: int = 0,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._runtime = runtime
        self._model = model
        self._chunk_max_chars = chunk_max_chars
        self._chunk_overlap_chars = chunk_overlap_chars

    async def ingest_directory(self, directory: Path) -> list[Chunk]:
        documents = load_markdown_directory(directory)
        return await self.ingest_documents(documents)

    async def ingest_documents(self, documents: list[Document]) -> list[Chunk]:
        chunks = chunk_documents(
            documents, max_chars=self._chunk_max_chars, overlap_chars=self._chunk_overlap_chars
        )
        if not chunks:
            return []
        vectors = await self._embedder.embed_documents([c.text for c in chunks])
        for chunk, vector in zip(chunks, vectors):
            await self._store.add(chunk.chunk_id, chunk.text, vector, metadata={"doc_id": chunk.doc_id})
        return chunks

    async def retrieve(self, question: str, k: int = 5) -> list[SearchResult]:
        query_embedding = await self._embedder.embed_query(question)
        return await self._store.search(query_embedding, k=k)

    async def chunk_count(self) -> int:
        return await self._store.count()

    async def answer(self, question: str, k: int = 5) -> RagAnswer:
        retrieved = await self.retrieve(question, k=k)
        prompt = build_rag_prompt(question, retrieved)
        request = LLMRequest(model=self._model, prompt=prompt)
        response = await self._runtime.generate(request)
        citations = extract_citations(response.text)
        return RagAnswer(
            question=question, answer_text=response.text, retrieved_chunks=retrieved, citations=citations
        )

"""Shared corpus/pipeline setup for Module 13's lab scripts -
`ScriptedGoldenRuntime` is a controlled stand-in for "a RAG generator,"
not a claim about what a real model would say: it returns each golden
case's own `expected_answer` (tagged with citation markers for its
`expected_source_ids`) when it recognizes that case's exact question, and
a small number of cases are deliberately corrupted (see
`CORRUPTED_QUESTION_IDS`) so this module's evaluation metrics have real
failures to catch, not just clean passes.
"""

from __future__ import annotations

from pathlib import Path

from local_ai_core.evals.golden_set import GoldenCase, load_golden_set
from local_ai_core.runtimes.types import LLMRequest, LLMResponse
from local_ai_rag.chunkers.document_chunker import chunk_documents
from local_ai_rag.embeddings.fake import FakeEmbedder
from local_ai_rag.loaders.markdown_loader import load_markdown_directory
from local_ai_rag.production_pipeline import ProductionRagPipeline
from local_ai_rag.stores.numpy_store import NumpyVectorStore

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CORPUS_DIR = REPO_ROOT / "datasets" / "rag_docs" / "nimbus_handbook"
GOLDEN_SET_PATH = REPO_ROOT / "evals" / "rag_eval" / "nimbus_golden_set.jsonl"

# q_003 (API rate limits): scripted to cite a document that was never
# retrieved - an invented citation (Module 11's "citations may be
# invented" gotcha, made real here).
# q_016 (unanswerable): scripted to answer confidently from "prior
# knowledge" instead of refusing (Module 11's other named gotcha).
CORRUPTED_QUESTION_IDS = {"q_003", "q_016"}


class ScriptedGoldenRuntime:
    def __init__(self, golden_cases: list[GoldenCase]) -> None:
        self._by_question = {case.question: case for case in golden_cases}

    async def generate(self, request: LLMRequest) -> LLMResponse:
        case = self._match_case(request.prompt)
        text = self._scripted_text(case)
        return LLMResponse(
            text=text,
            model=request.model,
            prompt_tokens=len(request.prompt.split()),
            completion_tokens=len(text.split()),
            latency_ms=0.0,
            stop_reason="stop",
        )

    def _match_case(self, prompt: str) -> GoldenCase | None:
        for question, case in self._by_question.items():
            if question in prompt:
                return case
        return None

    def _scripted_text(self, case: GoldenCase | None) -> str:
        if case is None:
            return "I don't know based on the provided documents."
        if case.question_id == "q_003":
            # Corrupted: cites a document never retrieved for this question
            # (verified empirically absent from this question's top-20 retrieval,
            # not just top-k, so this stays a real invented citation regardless
            # of which chunk of it might otherwise have surfaced).
            return self._with_citations(
                "The Nimbus API allows 10,000 requests per hour.", ["file_sharing::0"]
            )
        if case.question_id == "q_016":
            # Corrupted: confidently answers an unanswerable question instead of refusing.
            return "The Enterprise plan allows up to 500 workspace members."
        if case.requires_refusal:
            return case.expected_answer
        citation_ids = [f"{doc_id}::0" for doc_id in case.expected_source_ids]
        return self._with_citations(case.expected_answer, citation_ids)

    @staticmethod
    def _with_citations(sentence: str, citation_ids: list[str]) -> str:
        """Inserts citation markers *inside* the claim sentence, before its
        trailing period, not appended after it - citing text after a
        period starts a new "sentence" under simple punctuation-based
        splitting (`citation_verifier.py`'s `_sentence_containing`), which
        would silently disconnect the citation from the claim it supports.
        """
        markers = "".join(f" [{cid}]" for cid in citation_ids)
        if sentence.endswith("."):
            return f"{sentence[:-1]}{markers}."
        return f"{sentence}{markers}"

    async def stream(self, request: LLMRequest):  # pragma: no cover - not used by these labs
        raise NotImplementedError("ScriptedGoldenRuntime does not support streaming")

    async def tokenize(self, model: str, rendered_prompt: str) -> list[int]:
        return [abs(hash(word)) % 50_000 for word in rendered_prompt.split()]


async def build_pipeline_and_golden_set() -> tuple[ProductionRagPipeline, list[GoldenCase]]:
    documents = load_markdown_directory(CORPUS_DIR)
    embedder = FakeEmbedder(dimensions=64)
    store = NumpyVectorStore()
    chunks = chunk_documents(documents, max_chars=500)

    vectors = await embedder.embed_documents([c.text for c in chunks])
    for chunk, vector in zip(chunks, vectors):
        await store.add(chunk.chunk_id, chunk.text, vector, metadata={"doc_id": chunk.doc_id})

    golden_cases = load_golden_set(GOLDEN_SET_PATH)
    runtime = ScriptedGoldenRuntime(golden_cases)
    pipeline = ProductionRagPipeline(embedder, store, runtime, model="fake-model")
    return pipeline, golden_cases

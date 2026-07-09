from local_ai_rag.embeddings.embedder import SearchResult
from local_ai_rag.rerankers.cross_encoder_reranker import CrossEncoderReranker


def fake_load_fn(model_name: str):
    return f"loaded-model:{model_name}"


def fake_score_fn(model, pairs: list[tuple[str, str]]) -> list[float]:
    # Deterministic fake: score is how many words the pair shares.
    scores = []
    for query, text in pairs:
        query_words = set(query.lower().split())
        text_words = set(text.lower().split())
        scores.append(float(len(query_words & text_words)))
    return scores


def make_result(doc_id: str, text: str) -> SearchResult:
    return SearchResult(doc_id=doc_id, score=0.0, text=text, metadata={})


class TestRerank:
    async def test_reorders_candidates_by_the_score_fn(self):
        reranker = CrossEncoderReranker("fake-model", load_fn=fake_load_fn, score_fn=fake_score_fn)
        candidates = [make_result("low", "unrelated"), make_result("high", "reset password link")]
        results = await reranker.rerank("reset password", candidates)
        assert results[0].doc_id == "high"

    async def test_loads_the_model_only_once_across_calls(self):
        load_calls = []

        def counting_load_fn(model_name):
            load_calls.append(model_name)
            return fake_load_fn(model_name)

        reranker = CrossEncoderReranker("fake-model", load_fn=counting_load_fn, score_fn=fake_score_fn)
        candidates = [make_result("a", "text a")]
        await reranker.rerank("query", candidates)
        await reranker.rerank("query", candidates)
        assert load_calls == ["fake-model"]

    async def test_respects_k(self):
        reranker = CrossEncoderReranker("fake-model", load_fn=fake_load_fn, score_fn=fake_score_fn)
        candidates = [make_result(f"d{i}", f"text {i}") for i in range(5)]
        results = await reranker.rerank("text", candidates, k=2)
        assert len(results) == 2

    async def test_empty_candidates_returns_empty_list_without_loading_a_model(self):
        load_calls = []

        def counting_load_fn(model_name):
            load_calls.append(model_name)
            return fake_load_fn(model_name)

        reranker = CrossEncoderReranker("fake-model", load_fn=counting_load_fn, score_fn=fake_score_fn)
        results = await reranker.rerank("query", [])
        assert results == []
        assert load_calls == []

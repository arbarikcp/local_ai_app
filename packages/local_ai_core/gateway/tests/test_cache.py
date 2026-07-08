import numpy as np
import pytest

from local_ai_core.gateway.cache import (
    EmbeddingCache,
    ResponseCache,
    SemanticCache,
    embedding_cache_key,
    response_cache_key,
)


class TestResponseCacheKey:
    def test_same_inputs_produce_same_key(self):
        k1 = response_cache_key("m", "prompt", {"temp": 0.0}, "v1")
        k2 = response_cache_key("m", "prompt", {"temp": 0.0}, "v1")
        assert k1 == k2

    def test_different_model_produces_different_key(self):
        k1 = response_cache_key("model-a", "prompt", {}, "v1")
        k2 = response_cache_key("model-b", "prompt", {}, "v1")
        assert k1 != k2

    def test_different_prompt_produces_different_key(self):
        k1 = response_cache_key("m", "prompt A", {}, "v1")
        k2 = response_cache_key("m", "prompt B", {}, "v1")
        assert k1 != k2

    def test_different_prompt_version_produces_different_key(self):
        k1 = response_cache_key("m", "p", {}, "v1")
        k2 = response_cache_key("m", "p", {}, "v2")
        assert k1 != k2

    @pytest.mark.parametrize(
        "field",
        ["quantization", "tool_version", "schema_version", "safety_policy_version"],
    )
    def test_each_version_field_affects_the_key(self, field):
        # Gotcha (theory doc §11): cache keys must include these fields when
        # they affect output - verify each one actually changes the hash.
        base = response_cache_key("m", "p", {}, "v1")
        with_field = response_cache_key("m", "p", {}, "v1", **{field: "some-version"})
        assert base != with_field

    def test_different_params_produce_different_key(self):
        k1 = response_cache_key("m", "p", {"temperature": 0.0}, "v1")
        k2 = response_cache_key("m", "p", {"temperature": 0.7}, "v1")
        assert k1 != k2


class TestResponseCache:
    def test_miss_on_empty_cache(self):
        cache = ResponseCache()
        assert cache.get("some-key") is None
        assert cache.misses == 1
        assert cache.hits == 0

    def test_put_then_get_is_a_hit(self):
        cache = ResponseCache()
        cache.put("key1", "cached response")
        assert cache.get("key1") == "cached response"
        assert cache.hits == 1

    def test_hit_rate_computed_correctly(self):
        cache = ResponseCache()
        cache.put("key1", "value")
        cache.get("key1")  # hit
        cache.get("key1")  # hit
        cache.get("key2")  # miss
        assert cache.hit_rate == pytest.approx(2 / 3)

    def test_hit_rate_zero_with_no_activity(self):
        cache = ResponseCache()
        assert cache.hit_rate == 0.0

    def test_lru_eviction_when_over_capacity(self):
        cache = ResponseCache(max_entries=2)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)  # evicts "a" (least recently used)
        assert cache.get("a") is None
        assert cache.get("b") == 2
        assert cache.get("c") == 3

    def test_get_refreshes_lru_order(self):
        cache = ResponseCache(max_entries=2)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.get("a")  # "a" is now most-recently-used
        cache.put("c", 3)  # should evict "b", not "a"
        assert cache.get("a") == 1
        assert cache.get("b") is None

    def test_rejects_invalid_max_entries(self):
        with pytest.raises(ValueError):
            ResponseCache(max_entries=0)

    def test_len_reflects_entry_count(self):
        cache = ResponseCache()
        cache.put("a", 1)
        cache.put("b", 2)
        assert len(cache) == 2


class TestSemanticCache:
    def test_construction_rejects_invalid_threshold(self):
        with pytest.raises(ValueError):
            SemanticCache(similarity_threshold=1.5)
        with pytest.raises(ValueError):
            SemanticCache(similarity_threshold=-0.1)

    def test_miss_on_empty_cache(self):
        cache = SemanticCache()
        query = np.array([1.0, 0.0, 0.0])
        assert cache.get(query) is None
        assert cache.misses == 1

    def test_exact_match_embedding_is_a_hit(self):
        cache = SemanticCache(similarity_threshold=0.95)
        embedding = np.array([1.0, 0.0, 0.0])
        cache.put(embedding, "cached answer", original_query="What is X?")
        hit = cache.get(np.array([1.0, 0.0, 0.0]))
        assert hit is not None
        assert hit.response == "cached answer"
        assert hit.similarity == pytest.approx(1.0)
        assert hit.matched_query == "What is X?"

    def test_similar_but_not_identical_embedding_within_threshold_hits(self):
        cache = SemanticCache(similarity_threshold=0.9)
        cache.put(np.array([1.0, 0.0]), "answer")
        # A slightly rotated vector - similarity is high but not exactly 1.0.
        near_duplicate = np.array([0.99, 0.05])
        hit = cache.get(near_duplicate)
        assert hit is not None
        assert 0.9 <= hit.similarity < 1.0

    def test_dissimilar_embedding_is_a_miss(self):
        cache = SemanticCache(similarity_threshold=0.95)
        cache.put(np.array([1.0, 0.0]), "answer about topic A")
        unrelated = np.array([0.0, 1.0])  # orthogonal - similarity 0.0
        assert cache.get(unrelated) is None

    def test_returns_the_best_match_among_multiple_entries(self):
        cache = SemanticCache(similarity_threshold=0.5)
        cache.put(np.array([1.0, 0.0]), "answer A", original_query="query A")
        cache.put(np.array([0.9, 0.1]), "answer B", original_query="query B")
        hit = cache.get(np.array([1.0, 0.0]))
        assert hit.response == "answer A"

    def test_fifo_eviction_when_over_capacity(self):
        cache = SemanticCache(similarity_threshold=0.5, max_entries=2)
        cache.put(np.array([1.0, 0.0]), "first", original_query="q1")
        cache.put(np.array([0.0, 1.0]), "second", original_query="q2")
        cache.put(np.array([0.5, 0.5]), "third", original_query="q3")  # evicts "first"
        assert len(cache) == 2
        # "first"'s vector [1,0] is no longer present; searching for it should
        # now only match against "second"/"third", both dissimilar to [1,0].
        hit = cache.get(np.array([1.0, 0.0]))
        assert hit is None or hit.response != "first"

    def test_hit_rate_computed_correctly(self):
        cache = SemanticCache(similarity_threshold=0.95)
        cache.put(np.array([1.0, 0.0]), "answer")
        cache.get(np.array([1.0, 0.0]))  # hit
        cache.get(np.array([0.0, 1.0]))  # miss
        assert cache.hit_rate == pytest.approx(0.5)

    def test_zero_vector_query_does_not_crash(self):
        cache = SemanticCache()
        cache.put(np.array([1.0, 0.0]), "answer")
        result = cache.get(np.array([0.0, 0.0]))
        assert result is None  # cosine similarity undefined for zero vector -> treated as no match


class TestEmbeddingCacheKey:
    def test_same_inputs_produce_same_key(self):
        k1 = embedding_cache_key("hello world", "model-a")
        k2 = embedding_cache_key("hello world", "model-a")
        assert k1 == k2

    def test_different_model_produces_different_key(self):
        k1 = embedding_cache_key("text", "model-a")
        k2 = embedding_cache_key("text", "model-b")
        assert k1 != k2

    def test_different_normalization_version_produces_different_key(self):
        k1 = embedding_cache_key("text", "model-a", "v1")
        k2 = embedding_cache_key("text", "model-a", "v2")
        assert k1 != k2


class TestEmbeddingCache:
    def test_miss_then_put_then_hit(self):
        cache = EmbeddingCache()
        assert cache.get("hello", "model-a") is None
        embedding = np.array([0.1, 0.2, 0.3])
        cache.put("hello", "model-a", embedding)
        result = cache.get("hello", "model-a")
        assert np.array_equal(result, embedding)

    def test_different_embedding_model_is_a_separate_cache_entry(self):
        cache = EmbeddingCache()
        cache.put("hello", "model-a", np.array([1.0]))
        assert cache.get("hello", "model-b") is None  # not invalidated together, just distinct keys

    def test_lru_eviction_when_over_capacity(self):
        cache = EmbeddingCache(max_entries=2)
        cache.put("a", "m", np.array([1.0]))
        cache.put("b", "m", np.array([2.0]))
        cache.put("c", "m", np.array([3.0]))  # evicts "a"
        assert cache.get("a", "m") is None
        assert cache.get("c", "m") is not None

    def test_hit_rate_computed_correctly(self):
        cache = EmbeddingCache()
        cache.put("x", "m", np.array([1.0]))
        cache.get("x", "m")
        cache.get("y", "m")
        assert cache.hit_rate == pytest.approx(0.5)

    def test_rejects_invalid_max_entries(self):
        with pytest.raises(ValueError):
            EmbeddingCache(max_entries=0)

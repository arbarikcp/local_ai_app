import numpy as np
import pytest

from local_ai_rag.embeddings.embedder import (
    NumpyEmbeddingIndex,
    cosine_similarity,
    normalize,
    truncate_embedding,
)


class TestNormalize:
    def test_scales_to_unit_length(self):
        v = np.array([3.0, 4.0])  # 3-4-5 triangle
        result = normalize(v)
        assert np.linalg.norm(result) == pytest.approx(1.0)

    def test_zero_vector_is_returned_unchanged_not_a_crash(self):
        v = np.array([0.0, 0.0, 0.0])
        result = normalize(v)
        assert np.array_equal(result, v)

    def test_preserves_direction(self):
        v = np.array([1.0, 0.0])
        result = normalize(v)
        assert result[0] > 0 and result[1] == 0


class TestCosineSimilarity:
    def test_identical_vectors_have_similarity_one(self):
        v = np.array([1.0, 2.0, 3.0])
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors_have_similarity_zero(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors_have_similarity_negative_one(self):
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        assert cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_is_invariant_to_magnitude(self):
        a = np.array([1.0, 1.0])
        b = np.array([10.0, 10.0])
        assert cosine_similarity(a, b) == pytest.approx(1.0)

    def test_zero_vector_does_not_crash(self):
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 1.0])
        assert cosine_similarity(a, b) == pytest.approx(0.0)


class TestTruncateEmbedding:
    def test_truncates_to_requested_dimensions(self):
        v = normalize(np.array([1.0, 2.0, 3.0, 4.0]))
        result = truncate_embedding(v, 2)
        assert len(result) == 2

    def test_result_is_renormalized(self):
        v = normalize(np.array([1.0, 2.0, 3.0, 4.0]))
        result = truncate_embedding(v, 2)
        assert np.linalg.norm(result) == pytest.approx(1.0)

    def test_requesting_more_dimensions_than_available_returns_unchanged(self):
        v = np.array([1.0, 2.0])
        result = truncate_embedding(v, 10)
        assert np.array_equal(result, v)

    def test_rejects_nonpositive_dimensions(self):
        with pytest.raises(ValueError):
            truncate_embedding(np.array([1.0, 2.0]), 0)


class TestNumpyEmbeddingIndex:
    def test_len_reflects_number_of_added_documents(self):
        index = NumpyEmbeddingIndex()
        assert len(index) == 0
        index.add("d1", "text", np.array([1.0, 0.0]))
        assert len(index) == 1

    def test_contains_checks_doc_id_presence(self):
        index = NumpyEmbeddingIndex()
        index.add("d1", "text", np.array([1.0, 0.0]))
        assert "d1" in index
        assert "d2" not in index

    def test_search_returns_the_most_similar_document_first(self):
        index = NumpyEmbeddingIndex()
        index.add("close", "close doc", np.array([1.0, 0.1]))
        index.add("far", "far doc", np.array([0.0, 1.0]))
        results = index.search(np.array([1.0, 0.0]), k=2)
        assert results[0].doc_id == "close"
        assert results[0].score > results[1].score

    def test_search_respects_k(self):
        index = NumpyEmbeddingIndex()
        for i in range(5):
            index.add(f"d{i}", f"text {i}", np.array([1.0, float(i)]))
        results = index.search(np.array([1.0, 0.0]), k=3)
        assert len(results) == 3

    def test_search_rejects_nonpositive_k(self):
        index = NumpyEmbeddingIndex()
        with pytest.raises(ValueError):
            index.search(np.array([1.0]), k=0)

    def test_search_on_empty_index_returns_empty_list(self):
        index = NumpyEmbeddingIndex()
        assert index.search(np.array([1.0, 0.0])) == []

    def test_search_result_carries_text_and_metadata(self):
        index = NumpyEmbeddingIndex()
        index.add("d1", "the document text", np.array([1.0, 0.0]), metadata={"source": "manual"})
        results = index.search(np.array([1.0, 0.0]), k=1)
        assert results[0].text == "the document text"
        assert results[0].metadata == {"source": "manual"}

    def test_metadata_filter_excludes_nonmatching_documents(self):
        index = NumpyEmbeddingIndex()
        index.add("d1", "doc 1", np.array([1.0, 0.0]), metadata={"tenant": "a"})
        index.add("d2", "doc 2", np.array([1.0, 0.0]), metadata={"tenant": "b"})
        results = index.search(np.array([1.0, 0.0]), k=5, metadata_filter={"tenant": "a"})
        assert [r.doc_id for r in results] == ["d1"]

    def test_metadata_filter_requires_all_keys_to_match(self):
        index = NumpyEmbeddingIndex()
        index.add("d1", "doc 1", np.array([1.0, 0.0]), metadata={"tenant": "a", "lang": "en"})
        index.add("d2", "doc 2", np.array([1.0, 0.0]), metadata={"tenant": "a", "lang": "fr"})
        results = index.search(np.array([1.0, 0.0]), k=5, metadata_filter={"tenant": "a", "lang": "en"})
        assert [r.doc_id for r in results] == ["d1"]

    def test_adding_with_same_doc_id_overwrites(self):
        index = NumpyEmbeddingIndex()
        index.add("d1", "first version", np.array([1.0, 0.0]))
        index.add("d1", "second version", np.array([0.0, 1.0]))
        assert len(index) == 1
        results = index.search(np.array([0.0, 1.0]), k=1)
        assert results[0].text == "second version"

import pytest

from memory_math import (
    bytes_to_gb,
    bytes_to_gib,
    estimate_memory_budget,
    kv_cache_bytes,
    weights_bytes,
)


class TestWeightsBytes:
    def test_fp16_8b_model_matches_theory_doc_worked_example(self):
        # Theory doc §1: FP16 8B model ~= 16.0 GB (decimal), bits/param=16.0 => bytes = params * 2
        result_gb = bytes_to_gb(weights_bytes(8_000_000_000, "FP16"))
        assert result_gb == pytest.approx(16.0, rel=0.01)

    @pytest.mark.parametrize(
        "quant,expected_gb_for_8b",
        [
            ("FP16", 16.0),
            ("Q8_0", 8.5),
            ("Q6_K", 6.6),
            ("Q5_K_M", 5.7),
            ("Q4_K_M", 4.8),
            ("Q3_K_M", 3.9),
            ("Q2_K", 3.4),
        ],
    )
    def test_matches_theory_doc_rule_of_thumb_table(self, quant, expected_gb_for_8b):
        n_params = 8_000_000_000
        result_gb = bytes_to_gb(weights_bytes(n_params, quant))
        assert result_gb == pytest.approx(expected_gb_for_8b, rel=0.02)

    def test_lower_quant_means_less_memory(self):
        n_params = 7_000_000_000
        assert weights_bytes(n_params, "Q4_K_M") < weights_bytes(n_params, "Q8_0")
        assert weights_bytes(n_params, "Q8_0") < weights_bytes(n_params, "FP16")

    def test_unknown_quant_raises(self):
        with pytest.raises(ValueError):
            weights_bytes(1_000_000_000, "Q99_bogus")


class TestKvCacheBytes:
    def test_matches_theory_doc_worked_example_per_token_bytes(self):
        # Theory doc §4: 8B Llama-style (n_layers=32, n_kv_heads=8, head_dim=128)
        # => 128 KiB/token at FP16.
        per_token_bytes = kv_cache_bytes(
            n_layers=32, n_kv_heads=8, head_dim=128, context_tokens=1, kv_quant="FP16"
        )
        assert per_token_bytes == pytest.approx(128 * 1024, rel=0.001)

    @pytest.mark.parametrize(
        "context_tokens,expected_gib",
        [
            (4_096, 0.5),
            (8_192, 1.0),
            (32_768, 4.0),
            (128_000, 16.0 * (128_000 / 131_072)),  # table rounds 128K to 131072-token scale
        ],
    )
    def test_matches_theory_doc_context_table_fp16(self, context_tokens, expected_gib):
        result_gib = bytes_to_gib(
            kv_cache_bytes(n_layers=32, n_kv_heads=8, head_dim=128, context_tokens=context_tokens)
        )
        assert result_gib == pytest.approx(expected_gib, rel=0.05)

    def test_kv_quantization_halves_memory_relative_to_fp16(self):
        kwargs = dict(n_layers=32, n_kv_heads=8, head_dim=128, context_tokens=8192)
        fp16 = kv_cache_bytes(**kwargs, kv_quant="FP16")
        q8 = kv_cache_bytes(**kwargs, kv_quant="Q8_0")
        q4 = kv_cache_bytes(**kwargs, kv_quant="Q4_0")
        assert q8 == pytest.approx(fp16 / 2)
        assert q4 == pytest.approx(fp16 / 4)

    def test_concurrency_multiplies_kv_cache_linearly(self):
        kwargs = dict(n_layers=32, n_kv_heads=8, head_dim=128, context_tokens=8192)
        one = kv_cache_bytes(**kwargs, concurrent_sequences=1)
        four = kv_cache_bytes(**kwargs, concurrent_sequences=4)
        assert four == pytest.approx(one * 4)

    def test_grouped_query_attention_uses_kv_heads_not_full_heads(self):
        # A model with fewer KV heads (GQA) must produce a smaller cache than
        # one with more KV heads, all else equal - this is the whole point of GQA.
        gqa = kv_cache_bytes(n_layers=28, n_kv_heads=4, head_dim=128, context_tokens=8192)
        mha = kv_cache_bytes(n_layers=28, n_kv_heads=32, head_dim=128, context_tokens=8192)
        assert gqa < mha

    def test_unknown_kv_quant_raises(self):
        with pytest.raises(ValueError):
            kv_cache_bytes(n_layers=1, n_kv_heads=1, head_dim=1, context_tokens=1, kv_quant="bogus")

    def test_rejects_invalid_context_or_concurrency(self):
        with pytest.raises(ValueError):
            kv_cache_bytes(n_layers=1, n_kv_heads=1, head_dim=1, context_tokens=-1)
        with pytest.raises(ValueError):
            kv_cache_bytes(n_layers=1, n_kv_heads=1, head_dim=1, context_tokens=1, concurrent_sequences=0)


class TestEstimateMemoryBudget:
    def test_matches_theory_doc_8b_q4_8k_worked_example(self):
        # Theory doc §5: 4.8 GB weights + ~1.0 GiB KV cache + 0.5-1.5 GB overhead
        estimate = estimate_memory_budget(
            n_params=8_000_000_000,
            quant="Q4_K_M",
            n_layers=32,
            n_kv_heads=8,
            head_dim=128,
            context_tokens=8192,
        )
        assert estimate.weights_gb == pytest.approx(4.8, rel=0.02)
        assert estimate.kv_cache_gib == pytest.approx(1.0, rel=0.05)
        assert estimate.total_low_gib == pytest.approx(4.8 + 1.0 + 0.5, rel=0.05)
        assert estimate.total_high_gib == pytest.approx(4.8 + 1.0 + 1.5, rel=0.05)

    def test_total_high_always_exceeds_total_low(self):
        estimate = estimate_memory_budget(
            n_params=1_000_000_000, quant="Q4_K_M", n_layers=16, n_kv_heads=4, head_dim=64, context_tokens=2048
        )
        assert estimate.total_high_gib > estimate.total_low_gib

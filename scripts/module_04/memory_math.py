"""Weights + KV-cache + full-budget memory formulas.

Formulas match docs/modules/04_quantization_context_and_memory_math.md
exactly (§4-5) and are unit-tested against that doc's worked examples so the
notebook, the theory doc, and this code can never quietly drift apart.

These are *planning-grade estimates*, not exact allocator accounting — GGUF
k-quants store per-block scales, so bits-per-param is an effective average
(§1 of the theory doc). Lab 4.4 (predict, then measure) is what turns an
estimate into a verified claim.

Unit note: the curriculum bible's own tables are not internally consistent
about binary vs. decimal units — the weights "rule of thumb" table (§1) uses
decimal GB (1e9 bytes; that's why, e.g., an 8B model's bits-per-param number
and its GB rule-of-thumb number come out numerically equal), while the
KV-cache table (§4) explicitly uses binary GiB (1024^3 bytes, matching its
own "128 KiB/token" derivation). This module matches both source tables
exactly rather than silently "fixing" the inconsistency, and keeps the unit
in each field/function name so it's never ambiguous which one you're
reading.
"""

from __future__ import annotations

from dataclasses import dataclass

GIB = 1024**3
GB = 1_000_000_000

# Approximate effective bits per parameter, by quantization. Source: theory
# doc §1 / curriculum.md §14.
QUANT_BITS_PER_PARAM: dict[str, float] = {
    "FP16": 16.0,
    "Q8_0": 8.5,
    "Q6_K": 6.6,
    "Q5_K_M": 5.7,
    "Q4_K_M": 4.8,
    "Q3_K_M": 3.9,
    "Q2_K": 3.4,
}

# Bytes per KV-cache element, by KV-cache quantization. FP16 is the runtime
# default when a runtime doesn't support independent KV-cache quantization.
KV_QUANT_BYTES_PER_ELEMENT: dict[str, float] = {
    "FP16": 2.0,
    "Q8_0": 1.0,
    "Q4_0": 0.5,
}

# Planning-grade runtime overhead range, in GiB (theory doc §10 / §5 worked example).
RUNTIME_OVERHEAD_LOW_GIB = 0.5
RUNTIME_OVERHEAD_HIGH_GIB = 1.5


def weights_bytes(n_params: float, quant: str) -> float:
    """Approximate resident memory for model weights, in bytes.

    ``n_params`` is the raw parameter count (e.g. 8_000_000_000 for an 8B
    model), not billions.
    """
    if quant not in QUANT_BITS_PER_PARAM:
        raise ValueError(f"Unknown quantization: {quant!r}. Known: {sorted(QUANT_BITS_PER_PARAM)}")
    bits_per_param = QUANT_BITS_PER_PARAM[quant]
    return n_params * (bits_per_param / 8)


def kv_cache_bytes(
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    context_tokens: int,
    kv_quant: str = "FP16",
    concurrent_sequences: int = 1,
) -> float:
    """KV-cache memory in bytes.

    Mirrors: kv_bytes = 2 * n_layers * n_kv_heads * head_dim * context_tokens
                         * bytes_per_element(kv_quant) * concurrent_sequences
    Use n_kv_heads (grouped-query attention), not the full attention head count.
    """
    if kv_quant not in KV_QUANT_BYTES_PER_ELEMENT:
        raise ValueError(
            f"Unknown KV quantization: {kv_quant!r}. Known: {sorted(KV_QUANT_BYTES_PER_ELEMENT)}"
        )
    if context_tokens < 0 or concurrent_sequences < 1:
        raise ValueError("context_tokens must be >= 0 and concurrent_sequences must be >= 1")
    bytes_per_element = KV_QUANT_BYTES_PER_ELEMENT[kv_quant]
    elements_per_token = 2 * n_layers * n_kv_heads * head_dim
    return elements_per_token * context_tokens * bytes_per_element * concurrent_sequences


def bytes_to_gib(num_bytes: float) -> float:
    """Binary gibibytes (1024^3 bytes) — matches the theory doc's KV-cache table."""
    return num_bytes / GIB


def bytes_to_gb(num_bytes: float) -> float:
    """Decimal gigabytes (1e9 bytes) — matches the theory doc's weights rule-of-thumb table."""
    return num_bytes / GB


@dataclass(frozen=True)
class MemoryBudgetEstimate:
    """Mixed-unit by design (see module docstring): ``weights_gb`` is decimal
    GB, ``kv_cache_gib`` is binary GiB, matching each field to the curriculum
    table it reproduces. The two units differ by ~7.4%, which is well within
    the "planning-grade estimate" tolerance this whole module operates at —
    Lab 4.4 measures the real number.
    """

    weights_gb: float
    kv_cache_gib: float
    runtime_overhead_low_gib: float
    runtime_overhead_high_gib: float

    @property
    def total_low_gib(self) -> float:
        return self.weights_gb + self.kv_cache_gib + self.runtime_overhead_low_gib

    @property
    def total_high_gib(self) -> float:
        return self.weights_gb + self.kv_cache_gib + self.runtime_overhead_high_gib


def estimate_memory_budget(
    n_params: float,
    quant: str,
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    context_tokens: int,
    kv_quant: str = "FP16",
    concurrent_sequences: int = 1,
) -> MemoryBudgetEstimate:
    """Full planning-grade budget: weights + KV cache + runtime overhead range.

    Does NOT include app memory or OS memory (theory doc §5's full formula) —
    those are specific to what else is running and are out of scope for a
    model-level estimate.
    """
    w = bytes_to_gb(weights_bytes(n_params, quant))
    kv = bytes_to_gib(
        kv_cache_bytes(n_layers, n_kv_heads, head_dim, context_tokens, kv_quant, concurrent_sequences)
    )
    return MemoryBudgetEstimate(
        weights_gb=w,
        kv_cache_gib=kv,
        runtime_overhead_low_gib=RUNTIME_OVERHEAD_LOW_GIB,
        runtime_overhead_high_gib=RUNTIME_OVERHEAD_HIGH_GIB,
    )

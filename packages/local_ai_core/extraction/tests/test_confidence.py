from local_ai_core.extraction.confidence import ConfidenceInputs, compute_confidence, missing_required_fields


def _inputs(**overrides) -> ConfidenceInputs:
    defaults = dict(
        extracted_fields={"name": "Maria", "age": 29, "city": "Austin"},
        required_fields=["name", "age", "city"],
        used_repair_retry=False,
        used_constrained_decoding=True,
        had_conflicting_chunks=False,
    )
    defaults.update(overrides)
    return ConfidenceInputs(**defaults)


class TestComputeConfidence:
    def test_all_fields_present_no_risk_factors_is_high(self):
        assert compute_confidence(_inputs()) == "high"

    def test_one_missing_required_field_downgrades_to_medium(self):
        inputs = _inputs(extracted_fields={"name": "Maria", "age": None, "city": "Austin"})
        assert compute_confidence(inputs) == "medium"

    def test_repair_retry_alone_downgrades_to_medium(self):
        assert compute_confidence(_inputs(used_repair_retry=True)) == "medium"

    def test_unconstrained_decoding_alone_downgrades_to_medium(self):
        assert compute_confidence(_inputs(used_constrained_decoding=False)) == "medium"

    def test_conflicting_chunks_alone_downgrades_to_medium(self):
        assert compute_confidence(_inputs(had_conflicting_chunks=True)) == "medium"

    def test_two_risk_factors_is_low(self):
        inputs = _inputs(used_repair_retry=True, used_constrained_decoding=False)
        assert compute_confidence(inputs) == "low"

    def test_all_risk_factors_is_low_not_lower_than_low(self):
        inputs = _inputs(
            extracted_fields={"name": None, "age": None, "city": None},
            used_repair_retry=True,
            used_constrained_decoding=False,
            had_conflicting_chunks=True,
        )
        assert compute_confidence(inputs) == "low"

    def test_ignores_any_model_reported_confidence_field(self):
        # Even if the extracted_fields dict happens to contain a
        # "confidence" key (e.g. from InvoiceExtraction's schema), the
        # scorer must not read or trust it - only required_fields presence
        # and the pipeline facts matter.
        inputs = _inputs(
            extracted_fields={"name": "Maria", "age": 29, "city": "Austin", "confidence": "high"},
        )
        assert compute_confidence(inputs) == "high"
        low_but_claims_high = _inputs(
            extracted_fields={"name": None, "age": 29, "city": "Austin", "confidence": "high"},
            used_repair_retry=True,
        )
        assert compute_confidence(low_but_claims_high) == "low"


class TestMissingRequiredFields:
    def test_empty_when_nothing_missing(self):
        assert missing_required_fields(_inputs()) == []

    def test_lists_missing_field_names(self):
        inputs = _inputs(extracted_fields={"name": "Maria", "age": None, "city": None})
        assert missing_required_fields(inputs) == ["age", "city"]

    def test_ignores_fields_not_in_required_list(self):
        inputs = _inputs(
            extracted_fields={"name": "Maria", "age": 29, "city": "Austin", "extra": None},
            required_fields=["name", "age", "city"],
        )
        assert missing_required_fields(inputs) == []

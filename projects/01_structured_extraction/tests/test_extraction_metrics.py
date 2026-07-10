from extraction_metrics import field_exact_match, hallucinated_field_rate, missing_field_rate


class TestFieldExactMatch:
    def test_perfect_match_scores_one(self):
        reference = {"a": 1, "b": None}
        assert field_exact_match({"a": 1, "b": None}, reference) == 1.0

    def test_partial_match_scores_the_matched_fraction(self):
        reference = {"a": 1, "b": 2, "c": 3, "d": 4}
        predicted = {"a": 1, "b": 2, "c": 99, "d": 99}
        assert field_exact_match(predicted, reference) == 0.5

    def test_empty_reference_is_vacuously_one(self):
        assert field_exact_match({"a": 1}, {}) == 1.0

    def test_missing_key_in_predicted_counts_as_a_mismatch(self):
        reference = {"a": 1}
        assert field_exact_match({}, reference) == 0.0


class TestMissingFieldRate:
    def test_all_required_fields_present_scores_zero(self):
        reference = {"a": 1, "b": 2}
        assert missing_field_rate({"a": 1, "b": 2}, reference) == 0.0

    def test_a_missing_required_field_is_counted(self):
        reference = {"a": 1, "b": 2}
        assert missing_field_rate({"a": 1, "b": None}, reference) == 0.5

    def test_null_reference_fields_are_not_required(self):
        reference = {"a": None}
        assert missing_field_rate({}, reference) == 0.0


class TestHallucinatedFieldRate:
    def test_no_hallucination_scores_zero(self):
        reference = {"a": None}
        assert hallucinated_field_rate({"a": None}, reference) == 0.0

    def test_fabricating_a_value_for_a_null_reference_field_is_counted(self):
        reference = {"a": None, "b": None}
        assert hallucinated_field_rate({"a": "made up", "b": None}, reference) == 0.5

    def test_no_null_reference_fields_scores_zero(self):
        reference = {"a": 1}
        assert hallucinated_field_rate({"a": 1}, reference) == 0.0

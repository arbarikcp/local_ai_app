import pytest
from pydantic import ValidationError

from local_ai_core.extraction.schemas import InvoiceExtraction, PersonExtraction


class TestInvoiceExtraction:
    def test_valid_full_record_parses(self):
        record = InvoiceExtraction(
            invoice_number="A-4471",
            vendor_name="Globex Corp",
            invoice_date="2026-02-14",
            currency="USD",
            total_amount=1240.50,
            confidence="high",
            evidence={"invoice_number": "found in header"},
        )
        assert record.total_amount == pytest.approx(1240.50)

    def test_all_fields_except_confidence_are_optional(self):
        record = InvoiceExtraction(confidence="low")
        assert record.invoice_number is None
        assert record.vendor_name is None
        assert record.total_amount is None
        assert record.evidence == {}

    def test_confidence_is_required(self):
        with pytest.raises(ValidationError):
            InvoiceExtraction()

    def test_confidence_rejects_values_outside_the_literal(self):
        with pytest.raises(ValidationError):
            InvoiceExtraction(confidence="very high")

    def test_model_json_schema_is_a_valid_json_schema_object(self):
        schema = InvoiceExtraction.model_json_schema()
        assert schema["type"] == "object"
        assert "confidence" in schema["properties"]
        assert "confidence" in schema.get("required", [])


class TestPersonExtraction:
    def test_valid_full_record_parses(self):
        person = PersonExtraction(name="Maria", age=29, city="Austin")
        assert person.name == "Maria"
        assert person.age == 29

    def test_all_fields_are_optional(self):
        person = PersonExtraction()
        assert person.name is None
        assert person.age is None
        assert person.city is None

    def test_age_must_be_an_integer(self):
        with pytest.raises(ValidationError):
            PersonExtraction(age="twenty-nine")

    def test_model_json_schema_matches_field_names(self):
        schema = PersonExtraction.model_json_schema()
        assert set(schema["properties"].keys()) == {"name", "age", "city"}

from local_ai_core.tracing.pii_redaction import redact_pii


class TestEmailRedaction:
    def test_redacts_an_email_address(self):
        result = redact_pii("Contact me at jane.doe@example.com for details.")
        assert "jane.doe@example.com" not in result.redacted_text
        assert "[EMAIL]" in result.redacted_text
        assert result.redaction_counts["email"] == 1


class TestPhoneRedaction:
    def test_redacts_a_us_format_phone_number(self):
        result = redact_pii("Call me at 555-867-5309 tomorrow.")
        assert "555-867-5309" not in result.redacted_text
        assert "[PHONE]" in result.redacted_text
        assert result.redaction_counts["phone"] == 1


class TestSsnRedaction:
    def test_redacts_a_ssn_shaped_number(self):
        result = redact_pii("My SSN is 123-45-6789.")
        assert "123-45-6789" not in result.redacted_text
        assert "[SSN]" in result.redacted_text
        assert result.redaction_counts["ssn"] == 1


class TestCreditCardRedaction:
    def test_redacts_a_credit_card_shaped_number(self):
        result = redact_pii("My card number is 4111 1111 1111 1111.")
        assert "4111 1111 1111 1111" not in result.redacted_text
        assert "[CREDIT_CARD]" in result.redacted_text
        assert result.redaction_counts["credit_card"] == 1


class TestNoPiiPresent:
    def test_clean_text_is_unchanged_with_zero_redactions(self):
        result = redact_pii("The weather is nice today.")
        assert result.redacted_text == "The weather is nice today."
        assert result.total_redactions == 0
        assert result.redaction_counts == {}


class TestMultipleCategoriesInOneText:
    def test_redacts_every_category_present(self):
        text = "Email me at a@b.com or call 555-123-4567. SSN: 987-65-4321."
        result = redact_pii(text)
        assert "[EMAIL]" in result.redacted_text
        assert "[PHONE]" in result.redacted_text
        assert "[SSN]" in result.redacted_text
        assert result.total_redactions == 3

    def test_ssn_is_not_double_counted_as_a_phone_number(self):
        result = redact_pii("SSN: 987-65-4321.")
        assert "phone" not in result.redaction_counts
        assert result.redaction_counts["ssn"] == 1

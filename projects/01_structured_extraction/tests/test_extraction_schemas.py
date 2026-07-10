import pytest
from pydantic import ValidationError

from support_ticket_schema import SupportTicketExtraction


class TestSupportTicketExtraction:
    def test_accepts_a_fully_populated_ticket(self):
        ticket = SupportTicketExtraction.model_validate(
            {
                "category": "billing",
                "urgency": "high",
                "mentioned_product": "Personal Plan",
                "customer_email": "jane.doe@example.com",
                "summary": "Charged twice for renewal.",
            }
        )
        assert ticket.category == "billing"
        assert ticket.urgency == "high"

    def test_every_field_is_optional(self):
        ticket = SupportTicketExtraction.model_validate({})
        assert ticket.category is None
        assert ticket.urgency is None

    def test_category_is_constrained_to_the_known_taxonomy(self):
        with pytest.raises(ValidationError):
            SupportTicketExtraction.model_validate({"category": "not-a-real-category"})

    def test_urgency_is_constrained_to_the_known_levels(self):
        with pytest.raises(ValidationError):
            SupportTicketExtraction.model_validate({"urgency": "extremely-urgent"})

import pytest

from extraction_prompts import SCHEMA_REGISTRY, SchemaNotFoundError, get_prompt, resolve_schema
from local_ai_core.extraction.schemas import InvoiceExtraction
from support_ticket_schema import SupportTicketExtraction


class TestSchemaRegistry:
    def test_has_both_project_schemas(self):
        assert set(SCHEMA_REGISTRY.keys()) == {"invoice_v1", "support_ticket_v1"}

    def test_invoice_v1_maps_to_the_reused_module_8_schema(self):
        assert SCHEMA_REGISTRY["invoice_v1"].schema_class is InvoiceExtraction

    def test_support_ticket_v1_maps_to_the_new_schema(self):
        assert SCHEMA_REGISTRY["support_ticket_v1"].schema_class is SupportTicketExtraction


class TestResolveSchema:
    def test_resolves_a_known_schema_name(self):
        registered = resolve_schema("invoice_v1")
        assert registered.schema_name == "invoice_v1"
        assert registered.prompt_version == "v1"

    def test_unknown_schema_name_raises(self):
        with pytest.raises(SchemaNotFoundError):
            resolve_schema("does_not_exist_v1")


class TestGetPrompt:
    def test_builds_a_real_prompt_containing_the_schema_and_text(self):
        prompt, registered = get_prompt("invoice_v1", "Invoice #A-1 for $10.")
        assert "Invoice #A-1 for $10." in prompt
        assert "invoice_number" in prompt
        assert registered.schema_name == "invoice_v1"

    def test_support_ticket_prompt_contains_its_own_fields(self):
        prompt, registered = get_prompt("support_ticket_v1", "I was charged twice.")
        assert "category" in prompt
        assert "urgency" in prompt

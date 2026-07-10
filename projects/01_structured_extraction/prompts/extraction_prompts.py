"""Schema registry and prompt resolution (ARCHITECTURE.md "Data flow
through one request", step 3). Wraps Module 8's `build_extraction_prompt`
rather than reimplementing it, and records a `prompt_version` string per
schema - curriculum's own trace-model field (Module 21 theory doc §3,
"prompt template version") - so a future trace can tell exactly which
prompt template produced a given extraction.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel

from local_ai_core.extraction.pipeline import build_extraction_prompt
from local_ai_core.extraction.schemas import InvoiceExtraction
from support_ticket_schema import SupportTicketExtraction


class SchemaNotFoundError(Exception):
    def __init__(self, schema_name: str) -> None:
        super().__init__(f"no extraction schema registered under {schema_name!r}")
        self.schema_name = schema_name


@dataclass(frozen=True)
class RegisteredSchema:
    schema_name: str
    schema_class: type[BaseModel]
    prompt_version: str


SCHEMA_REGISTRY: dict[str, RegisteredSchema] = {
    "invoice_v1": RegisteredSchema(schema_name="invoice_v1", schema_class=InvoiceExtraction, prompt_version="v1"),
    "support_ticket_v1": RegisteredSchema(
        schema_name="support_ticket_v1", schema_class=SupportTicketExtraction, prompt_version="v1"
    ),
}


def resolve_schema(schema_name: str) -> RegisteredSchema:
    registered = SCHEMA_REGISTRY.get(schema_name)
    if registered is None:
        raise SchemaNotFoundError(schema_name)
    return registered


def get_prompt(schema_name: str, text: str) -> tuple[str, RegisteredSchema]:
    registered = resolve_schema(schema_name)
    prompt = build_extraction_prompt(text, registered.schema_class)
    return prompt, registered

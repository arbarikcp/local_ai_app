"""Lab 1 — five prompt variants for the same extraction task, at increasing
discipline levels (theory doc "Prompt design principles"). Variant 5 is the
curriculum's own example extraction prompt, plus a few-shot example.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "packages"))

from local_ai_core.prompts.few_shot import FewShotExample  # noqa: E402
from local_ai_core.prompts.template import PromptTemplate  # noqa: E402

EXTRACTION_SCHEMA_DESCRIPTION = '{"name": string or null, "age": integer or null, "city": string or null}'


def variant_1_vague(text: str) -> str:
    """Level 1: vague, no structure - the anti-pattern this module argues against."""
    return f"Get the important info from this: {text}"


def variant_2_direct_task() -> PromptTemplate:
    """Level 2: a direct task statement, no rules, no schema."""
    return PromptTemplate(
        prompt_id="extraction",
        version="v2-direct-task",
        role="You extract information from text.",
        task="Extract the name, age, and city mentioned in the input.",
    )


def variant_3_with_rules() -> PromptTemplate:
    """Level 3: adds explicit output rules."""
    return PromptTemplate(
        prompt_id="extraction",
        version="v3-with-rules",
        role="You are an information extraction engine.",
        task="Extract the name, age, and city mentioned in the input.",
        rules=[
            "Return only valid JSON.",
            "Do not include markdown.",
            "If a field is missing, use null.",
        ],
    )


def variant_4_with_schema() -> PromptTemplate:
    """Level 4: adds an explicit output schema (the curriculum's example prompt)."""
    return PromptTemplate(
        prompt_id="extraction",
        version="v4-with-schema",
        role="You are an information extraction engine.",
        task="Extract the requested fields from the input text.",
        output_contract=f"Strict JSON matching this schema: {EXTRACTION_SCHEMA_DESCRIPTION}",
        rules=[
            "Return only valid JSON.",
            "Do not include markdown.",
            "If a field is missing, use null.",
            "Do not infer values that are not present.",
            "Follow the schema exactly.",
        ],
    )


def variant_5_with_few_shot() -> PromptTemplate:
    """Level 5: variant 4 plus a few-shot example - the most disciplined variant."""
    base = variant_4_with_schema()
    return PromptTemplate(
        prompt_id="extraction",
        version="v5-with-few-shot",
        role=base.role,
        task=base.task,
        output_contract=base.output_contract,
        rules=base.rules,
        few_shot_examples=[
            FewShotExample(
                input="Bob moved to Denver two years ago. He is 41.",
                output='{"name": "Bob", "age": 41, "city": "Denver"}',
            )
        ],
    )


def variant_4_compressed() -> PromptTemplate:
    """A deliberately compressed version of variant 4 - fewer, terser rules,
    no schema prose - for Lab 6's compression-vs-quality comparison. Not a
    production compression algorithm, just a controlled point of comparison
    against variant_4_with_schema().
    """
    return PromptTemplate(
        prompt_id="extraction",
        version="v4-compressed",
        role="Extraction engine.",
        task="Extract name/age/city as JSON.",
        rules=["JSON only.", "null if missing."],
    )


ALL_VARIANTS = {
    "v1_vague": variant_1_vague,
    "v2_direct_task": variant_2_direct_task,
    "v3_with_rules": variant_3_with_rules,
    "v4_with_schema": variant_4_with_schema,
    "v5_with_few_shot": variant_5_with_few_shot,
}


def render_variant(name: str, text: str) -> str:
    """Render any variant (function-based or PromptTemplate-based) into a
    final prompt string, uniformly.
    """
    variant = ALL_VARIANTS[name]
    if name == "v1_vague":
        return variant(text)
    template = variant()
    return template.render(text)

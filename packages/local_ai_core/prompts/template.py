"""PromptTemplate — assembles the canonical Role/Task/Input contract/Output
contract/Rules/Examples/User input structure (theory doc "Prompt template
structure").

Invariant sections (everything except the final user input) are rendered
first, variable content last - this is also Module 6.5's prompt-prefix-
reuse layout rule, so this structure and that caching rule reinforce each
other rather than being two conventions to remember separately.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .few_shot import FewShotExample, NegativeExample, format_few_shot_examples, format_negative_examples


@dataclass(frozen=True)
class PromptTemplate:
    prompt_id: str
    version: str
    role: str
    task: str
    input_contract: str = ""
    output_contract: str = ""
    rules: list[str] = field(default_factory=list)
    few_shot_examples: list[FewShotExample] = field(default_factory=list)
    negative_examples: list[NegativeExample] = field(default_factory=list)

    def render(self, user_input: str) -> str:
        """Render the full prompt, invariant sections first, user input last."""
        sections: list[str] = []

        if self.role:
            sections.append(self.role)
        if self.task:
            sections.append(f"Task:\n{self.task}")
        if self.input_contract:
            sections.append(f"Input contract:\n{self.input_contract}")
        if self.output_contract:
            sections.append(f"Output contract:\n{self.output_contract}")
        if self.rules:
            rules_text = "\n".join(f"- {rule}" for rule in self.rules)
            sections.append(f"Rules:\n{rules_text}")

        examples_text = self._render_examples()
        if examples_text:
            sections.append(f"Examples:\n{examples_text}")

        sections.append(f"Input:\n{user_input}")
        return "\n\n".join(sections)

    def _render_examples(self) -> str:
        parts = []
        positive = format_few_shot_examples(self.few_shot_examples)
        if positive:
            parts.append(positive)
        negative = format_negative_examples(self.negative_examples)
        if negative:
            parts.append(negative)
        return "\n\n".join(parts)

    def invariant_prefix(self) -> str:
        """Everything except the final user input - the part a caching/
        prompt-prefix-reuse layer should treat as stable (Module 6.5 §7-9).
        """
        return self.render("").rsplit("Input:\n", 1)[0].rstrip()

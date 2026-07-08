import pytest

import prompt_variants as pv


class TestIndividualVariants:
    def test_variant_1_is_a_plain_string_containing_the_input(self):
        result = pv.variant_1_vague("Maria is 29.")
        assert isinstance(result, str)
        assert "Maria is 29." in result

    def test_variant_2_has_no_rules_or_schema(self):
        template = pv.variant_2_direct_task()
        assert template.rules == []
        assert template.output_contract == ""

    def test_variant_3_has_rules_but_no_schema(self):
        template = pv.variant_3_with_rules()
        assert len(template.rules) > 0
        assert template.output_contract == ""

    def test_variant_4_has_rules_and_schema(self):
        template = pv.variant_4_with_schema()
        assert len(template.rules) > 0
        assert template.output_contract != ""
        assert pv.EXTRACTION_SCHEMA_DESCRIPTION in template.output_contract

    def test_variant_5_has_everything_variant_4_has_plus_a_few_shot_example(self):
        v4 = pv.variant_4_with_schema()
        v5 = pv.variant_5_with_few_shot()
        assert v5.rules == v4.rules
        assert v5.output_contract == v4.output_contract
        assert len(v5.few_shot_examples) == 1

    def test_variant_4_compressed_is_shorter_than_variant_4(self):
        full = pv.variant_4_with_schema()
        compressed = pv.variant_4_compressed()
        assert len(compressed.invariant_prefix()) < len(full.invariant_prefix())


class TestDisciplineIsMonotonicallyIncreasing:
    def test_each_variant_is_at_least_as_long_as_the_previous_except_vague(self):
        # variant 1 (vague) is a different shape entirely (plain string, no
        # structure) - compare variants 2-5, which should grow monotonically
        # as rules/schema/examples are added.
        lengths = [
            len(pv.variant_2_direct_task().invariant_prefix()),
            len(pv.variant_3_with_rules().invariant_prefix()),
            len(pv.variant_4_with_schema().invariant_prefix()),
            len(pv.variant_5_with_few_shot().invariant_prefix()),
        ]
        assert lengths == sorted(lengths)
        assert lengths[0] < lengths[-1]


class TestRenderVariant:
    @pytest.mark.parametrize("name", list(pv.ALL_VARIANTS.keys()))
    def test_every_variant_renders_a_nonempty_prompt_containing_the_input(self, name):
        rendered = pv.render_variant(name, "Test input text.")
        assert isinstance(rendered, str)
        assert len(rendered) > 0
        assert "Test input text." in rendered

    def test_unknown_variant_name_raises_key_error(self):
        with pytest.raises(KeyError):
            pv.render_variant("not-a-real-variant", "text")

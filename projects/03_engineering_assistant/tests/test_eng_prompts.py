from eng_prompts import build_explain_symbol_prompt, build_generate_tests_prompt, build_suggest_refactor_prompt


class TestBuildExplainSymbolPrompt:
    def test_includes_the_real_symbol_name_and_source(self):
        prompt = build_explain_symbol_prompt("remove_stock", "def remove_stock(...): ...")
        assert "remove_stock" in prompt
        assert "def remove_stock(...): ..." in prompt
        assert "Explain what this symbol does" in prompt


class TestBuildSuggestRefactorPrompt:
    def test_includes_the_real_source_and_does_not_ask_for_a_patch(self):
        prompt = build_suggest_refactor_prompt("def foo(): pass")
        assert "def foo(): pass" in prompt
        assert "Do not propose a patch" in prompt


class TestBuildGenerateTestsPrompt:
    def test_includes_symbol_and_source_and_asks_for_only_code(self):
        prompt = build_generate_tests_prompt("calculate_discount", "def calculate_discount(...): ...")
        assert "calculate_discount" in prompt
        assert "Respond with only the test function's source code" in prompt

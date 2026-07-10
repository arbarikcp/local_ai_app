from eng_intent_classifier import IntentType, classify_intent


class TestEachIntentType:
    def test_generate_tests(self):
        result = classify_intent("Please generate tests for the stock module.")
        assert result.intent == IntentType.GENERATE_TESTS

    def test_run_tests(self):
        result = classify_intent("Run the tests please.")
        assert result.intent == IntentType.RUN_TESTS

    def test_suggest_refactor(self):
        result = classify_intent("Can you refactor this function?")
        assert result.intent == IntentType.SUGGEST_REFACTOR

    def test_propose_patch(self):
        result = classify_intent("There's a bug in remove_stock, please fix it.")
        assert result.intent == IntentType.PROPOSE_PATCH

    def test_explain_symbol(self):
        result = classify_intent("Explain function remove_stock.")
        assert result.intent == IntentType.EXPLAIN_SYMBOL

    def test_search_code(self):
        result = classify_intent("Search for the word discount in the repo.")
        assert result.intent == IntentType.SEARCH_CODE

    def test_explain_repo(self):
        result = classify_intent("Give me an overview of this repo's structure.")
        assert result.intent == IntentType.EXPLAIN_REPO


class TestOrderingSensitiveCases:
    def test_add_test_is_not_misclassified_as_run_tests(self):
        # "add test coverage" contains "test" but must not match RUN_TESTS's
        # "run test" keyword - GENERATE_TESTS is checked first.
        result = classify_intent("Please add test coverage for the pricing module.")
        assert result.intent == IntentType.GENERATE_TESTS


class TestDefaultFallback:
    def test_an_unrecognized_request_defaults_to_explain_repo(self):
        result = classify_intent("asdkjfh qwoeiru")
        assert result.intent == IntentType.EXPLAIN_REPO
        assert "no keyword matched" in result.reason


class TestReasonIsTraceable:
    def test_every_classification_carries_a_nonempty_reason(self):
        for request in ["fix the bug", "search for x", "asdkjfh"]:
            result = classify_intent(request)
            assert len(result.reason) > 0

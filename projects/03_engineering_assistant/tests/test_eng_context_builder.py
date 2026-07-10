from pathlib import Path

import pytest

from eng_context_builder import SymbolNotFoundError, build_context
from eng_intent_classifier import IntentType

DEMO_REPO = Path(__file__).resolve().parent.parent / "demo_repo"


class TestExplainRepo:
    def test_lists_symbols_across_every_real_file(self):
        bundle = build_context(DEMO_REPO, IntentType.EXPLAIN_REPO)
        assert "inventory/stock.py" in bundle.repo_symbols
        assert "inventory/pricing.py" in bundle.repo_symbols
        assert "inventory/reports.py" in bundle.repo_symbols

    def test_finds_the_real_remove_stock_function(self):
        bundle = build_context(DEMO_REPO, IntentType.EXPLAIN_REPO)
        stock_symbols = bundle.repo_symbols["inventory/stock.py"]
        assert any(s.name == "remove_stock" for s in stock_symbols)


class TestSearchCode:
    def test_finds_real_matches_for_a_real_query(self):
        bundle = build_context(DEMO_REPO, IntentType.SEARCH_CODE, query="quantity")
        assert len(bundle.search_results) > 0
        assert all("quantity" in m.line_text.lower() for m in bundle.search_results)

    def test_missing_query_raises(self):
        with pytest.raises(ValueError):
            build_context(DEMO_REPO, IntentType.SEARCH_CODE)


class TestExplainSymbol:
    def test_locates_the_real_function_and_its_source_excerpt(self):
        bundle = build_context(DEMO_REPO, IntentType.EXPLAIN_SYMBOL, symbol_name="remove_stock")
        assert "inventory/stock.py" in bundle.repo_symbols
        assert bundle.file_excerpt is not None
        assert "def remove_stock" in bundle.file_excerpt

    def test_unknown_symbol_raises(self):
        with pytest.raises(SymbolNotFoundError):
            build_context(DEMO_REPO, IntentType.EXPLAIN_SYMBOL, symbol_name="does_not_exist_anywhere")

    def test_target_file_alone_returns_the_whole_files_symbols_and_excerpt(self):
        bundle = build_context(DEMO_REPO, IntentType.EXPLAIN_SYMBOL, target_file="inventory/pricing.py")
        assert any(s.name == "calculate_discount" for s in bundle.repo_symbols["inventory/pricing.py"])
        assert "def calculate_discount" in bundle.file_excerpt

    def test_neither_symbol_name_nor_target_file_raises(self):
        with pytest.raises(ValueError):
            build_context(DEMO_REPO, IntentType.EXPLAIN_SYMBOL)


class TestRunTests:
    def test_needs_no_context(self):
        bundle = build_context(DEMO_REPO, IntentType.RUN_TESTS)
        assert bundle.repo_symbols == {}
        assert bundle.search_results == []
        assert bundle.file_excerpt is None

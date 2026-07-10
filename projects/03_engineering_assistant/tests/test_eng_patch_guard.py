import pytest

from eng_patch_guard import PatchLineCountError, PatchScopeError, validate_hunk_line_counts, validate_patch_scope
from local_ai_agents.tools.patch_tools import validate_patch_format

VALID_PATCH = """--- inventory/stock.py
+++ inventory/stock.py
@@ -20,3 +20,5 @@
 def remove_stock(item: StockItem, amount: int) -> StockItem:
+    if amount > item.quantity:
+        raise ValueError("cannot remove more than current quantity")
     item.quantity -= amount
     return item
"""

WRONG_OLD_COUNT_PATCH = """--- inventory/stock.py
+++ inventory/stock.py
@@ -20,99 +20,5 @@
 def remove_stock(item: StockItem, amount: int) -> StockItem:
+    if amount > item.quantity:
+        raise ValueError("cannot remove more than current quantity")
     item.quantity -= amount
     return item
"""

WRONG_NEW_COUNT_PATCH = """--- inventory/stock.py
+++ inventory/stock.py
@@ -20,3 +20,99 @@
 def remove_stock(item: StockItem, amount: int) -> StockItem:
+    if amount > item.quantity:
+        raise ValueError("cannot remove more than current quantity")
     item.quantity -= amount
     return item
"""

NO_COUNT_PATCH = """--- inventory/stock.py
+++ inventory/stock.py
@@ -20 +20 @@
 def remove_stock(item: StockItem, amount: int) -> StockItem:
"""


class TestValidatePatchScope:
    def test_a_patch_targeting_the_expected_file_passes(self):
        parsed = validate_patch_format(VALID_PATCH)
        validate_patch_scope(parsed, "inventory/stock.py")  # should not raise

    def test_a_patch_targeting_a_different_file_is_rejected(self):
        parsed = validate_patch_format(VALID_PATCH)
        with pytest.raises(PatchScopeError):
            validate_patch_scope(parsed, "inventory/pricing.py")

    def test_the_error_names_both_files(self):
        parsed = validate_patch_format(VALID_PATCH)
        with pytest.raises(PatchScopeError) as exc_info:
            validate_patch_scope(parsed, "inventory/pricing.py")
        assert exc_info.value.actual_file_path == "inventory/stock.py"
        assert exc_info.value.expected_file_path == "inventory/pricing.py"


class TestValidateHunkLineCounts:
    def test_a_patch_with_correct_counts_passes(self):
        validate_hunk_line_counts(VALID_PATCH)  # should not raise

    def test_a_wrong_old_count_is_rejected(self):
        with pytest.raises(PatchLineCountError):
            validate_hunk_line_counts(WRONG_OLD_COUNT_PATCH)

    def test_a_wrong_new_count_is_rejected(self):
        with pytest.raises(PatchLineCountError):
            validate_hunk_line_counts(WRONG_NEW_COUNT_PATCH)

    def test_an_omitted_count_is_not_checked(self):
        # Bare "@@ -20 +20 @@" (no ",N") is valid unified-diff shorthand
        # for a 1-line hunk - nothing to check since no count was claimed.
        validate_hunk_line_counts(NO_COUNT_PATCH)  # should not raise

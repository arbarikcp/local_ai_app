import pytest

from inventory.reports import generate_summary
from inventory.stock import StockItem


def test_generate_summary_computes_subtotal_and_tax():
    items = [
        StockItem(name="widget", quantity=10, unit_price=2.0),
        StockItem(name="gadget", quantity=3, unit_price=5.0),
    ]
    summary = generate_summary(items, tax_rate=0.10)
    assert summary["item_count"] == 2
    assert summary["subtotal"] == 35.0
    assert summary["total_with_tax"] == pytest.approx(38.5)


def test_generate_summary_with_no_items():
    summary = generate_summary([], tax_rate=0.10)
    assert summary["item_count"] == 0
    assert summary["subtotal"] == 0.0
    assert summary["total_with_tax"] == 0.0

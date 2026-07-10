import pytest

from inventory.stock import StockItem, add_stock, remove_stock, total_value


def test_add_stock_increases_quantity():
    item = StockItem(name="widget", quantity=10, unit_price=2.0)
    add_stock(item, 5)
    assert item.quantity == 15


def test_add_stock_rejects_negative_amount():
    item = StockItem(name="widget", quantity=10, unit_price=2.0)
    with pytest.raises(ValueError):
        add_stock(item, -1)


def test_remove_stock_decreases_quantity():
    item = StockItem(name="widget", quantity=10, unit_price=2.0)
    remove_stock(item, 4)
    assert item.quantity == 6


def test_remove_stock_more_than_available_raises_value_error():
    # Real, currently-failing test: remove_stock() has no validation
    # against removing more than the current quantity, so this currently
    # goes negative instead of raising.
    item = StockItem(name="widget", quantity=5, unit_price=2.0)
    with pytest.raises(ValueError):
        remove_stock(item, 10)


def test_total_value_sums_quantity_times_price():
    items = [
        StockItem(name="widget", quantity=10, unit_price=2.0),
        StockItem(name="gadget", quantity=3, unit_price=5.0),
    ]
    assert total_value(items) == 35.0

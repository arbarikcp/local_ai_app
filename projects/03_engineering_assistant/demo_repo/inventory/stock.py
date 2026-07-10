"""Stock tracking for the inventory demo repo."""

from dataclasses import dataclass


@dataclass
class StockItem:
    name: str
    quantity: int
    unit_price: float


def add_stock(item: StockItem, amount: int) -> StockItem:
    if amount < 0:
        raise ValueError("amount must be non-negative")
    item.quantity += amount
    return item


def remove_stock(item: StockItem, amount: int) -> StockItem:
    item.quantity -= amount
    return item


def total_value(items: list[StockItem]) -> float:
    return sum(item.quantity * item.unit_price for item in items)

"""Summary reporting for the inventory demo repo - a real cross-file
dependency, importing from both stock.py and pricing.py.
"""

from inventory.pricing import apply_tax
from inventory.stock import StockItem, total_value


def generate_summary(items: list[StockItem], tax_rate: float = 0.0) -> dict:
    subtotal = total_value(items)
    return {
        "item_count": len(items),
        "subtotal": subtotal,
        "total_with_tax": apply_tax(subtotal, tax_rate),
    }

"""Pricing calculations for the inventory demo repo."""


def calculate_discount(price: float, percent_off: float) -> float:
    if not 0 <= percent_off <= 100:
        raise ValueError("percent_off must be between 0 and 100")
    return price * (1 - percent_off / 100)


def apply_tax(price: float, tax_rate: float) -> float:
    if tax_rate < 0:
        raise ValueError("tax_rate must be non-negative")
    return price * (1 + tax_rate)

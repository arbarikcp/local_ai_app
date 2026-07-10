import pytest

from inventory.pricing import apply_tax, calculate_discount


def test_calculate_discount():
    assert calculate_discount(100.0, 20) == 80.0


def test_calculate_discount_rejects_out_of_range_percent():
    with pytest.raises(ValueError):
        calculate_discount(100.0, 150)


def test_apply_tax():
    assert apply_tax(100.0, 0.08) == pytest.approx(108.0)


def test_apply_tax_rejects_negative_rate():
    with pytest.raises(ValueError):
        apply_tax(100.0, -0.1)

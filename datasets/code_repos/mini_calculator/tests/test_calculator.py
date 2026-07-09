import pytest

from calculator import add, average, divide, multiply, subtract


def test_add():
    assert add(2, 3) == 5


def test_subtract():
    assert subtract(5, 3) == 2


def test_multiply():
    assert multiply(4, 5) == 20


def test_divide():
    assert divide(10, 2) == 5


def test_divide_by_zero_raises():
    with pytest.raises(ValueError):
        divide(1, 0)


def test_average_of_nonempty_list():
    assert average([1, 2, 3]) == 2


def test_average_of_empty_list_should_return_zero():
    assert average([]) == 0

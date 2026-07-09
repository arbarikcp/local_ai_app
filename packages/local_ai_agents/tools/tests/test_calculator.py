import pytest

from local_ai_agents.tools.calculator import CalculatorArgs, UnsafeExpressionError, calculator_handler, safe_eval


class TestSafeEval:
    def test_basic_arithmetic(self):
        assert safe_eval("2 + 2") == 4

    def test_operator_precedence(self):
        assert safe_eval("2 + 3 * 4") == 14

    def test_parentheses(self):
        assert safe_eval("(2 + 3) * 4") == 20

    def test_exponentiation(self):
        assert safe_eval("2 ** 10") == 1024

    def test_unary_negative(self):
        assert safe_eval("-5 + 10") == 5

    def test_floor_division_and_modulo(self):
        assert safe_eval("7 // 2") == 3
        assert safe_eval("7 % 2") == 1

    def test_division_by_zero_raises_a_real_zero_division_error(self):
        with pytest.raises(ZeroDivisionError):
            safe_eval("1 / 0")


class TestSafeEvalRejectsUnsafeInput:
    def test_rejects_function_calls(self):
        with pytest.raises(UnsafeExpressionError):
            safe_eval("__import__('os').system('echo pwned')")

    def test_rejects_name_references(self):
        with pytest.raises(UnsafeExpressionError):
            safe_eval("os.system('ls')")

    def test_rejects_attribute_access(self):
        with pytest.raises(UnsafeExpressionError):
            safe_eval("(1).__class__")

    def test_rejects_string_literals(self):
        with pytest.raises(UnsafeExpressionError):
            safe_eval("'a' + 'b'")

    def test_rejects_boolean_literals(self):
        with pytest.raises(UnsafeExpressionError):
            safe_eval("True")

    def test_rejects_invalid_syntax(self):
        with pytest.raises(UnsafeExpressionError):
            safe_eval("2 +")

    def test_rejects_list_literals(self):
        with pytest.raises(UnsafeExpressionError):
            safe_eval("[1, 2, 3]")


class TestCalculatorHandler:
    async def test_returns_the_evaluated_result(self):
        result = await calculator_handler(CalculatorArgs(expression="6 * 7"))
        assert result == 42

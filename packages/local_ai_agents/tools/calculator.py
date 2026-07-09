"""Safe calculator tool (Lab 2) - real safety via AST whitelisting, not
Python `eval()`. Only numeric literals, `+ - * / ** % //`, unary +/-, and
parentheses are permitted; any other AST node (`Name`, `Call`,
`Attribute`, `Subscript`, ...) raises before any evaluation happens - so
`__import__('os').system(...)`-style payloads never reach the interpreter.
"""

from __future__ import annotations

import ast
import operator

from pydantic import BaseModel, Field

from local_ai_agents.tools.base import Tool


class CalculatorArgs(BaseModel):
    expression: str = Field(min_length=1, max_length=200)


class UnsafeExpressionError(Exception):
    pass


_BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}

_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise UnsafeExpressionError(f"Non-numeric constant: {node.value!r}")
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
        return _BINARY_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    raise UnsafeExpressionError(f"Disallowed expression element: {type(node).__name__}")


def safe_eval(expression: str) -> float:
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise UnsafeExpressionError(f"Could not parse expression: {expression!r}") from exc
    return _eval_node(tree)


async def calculator_handler(args: CalculatorArgs) -> float:
    return safe_eval(args.expression)


calculator_tool = Tool(
    name="calculator",
    description="Evaluate a numeric arithmetic expression (+, -, *, /, **, %, //, parentheses only).",
    args_model=CalculatorArgs,
    handler=calculator_handler,
)

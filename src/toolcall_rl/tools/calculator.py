"""Calculator tool."""

from __future__ import annotations

import ast
import operator


_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def calculator(expression: str) -> dict[str, str | float | int]:
    """Evaluate a basic arithmetic expression.

    Supports numbers, parentheses, and these operators: +, -, *, /, //, %, **.
    """

    try:
        parsed_expression = ast.parse(expression, mode="eval")
        result = _evaluate_math_node(parsed_expression.body)
    except Exception as exc:
        return {
            "expression": expression,
            "error": f"Could not calculate expression: {exc}",
        }

    return {
        "expression": expression,
        "result": result,
    }


def _evaluate_math_node(node: ast.AST) -> int | float:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return node.value

    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _evaluate_math_node(node.left)
        right = _evaluate_math_node(node.right)
        return _BINARY_OPERATORS[type(node.op)](left, right)

    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        operand = _evaluate_math_node(node.operand)
        return _UNARY_OPERATORS[type(node.op)](operand)

    raise ValueError(f"unsupported expression element: {type(node).__name__}")

from __future__ import annotations

import ast
from decimal import Decimal, DivisionByZero, InvalidOperation, localcontext
from typing import Any

_ALLOWED_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Constant,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.Mod,
    ast.USub,
    ast.UAdd,
)
_MAX_PRECISION = 100
_MAX_EXPONENT = 1000
_MAX_RESULT_DIGITS = 10_000


class CalculatorError(ValueError):
    pass


def _format(value: Decimal, precision: int, notation: str) -> str:
    if not value.is_finite():
        raise CalculatorError("计算结果不是有限数值")
    if notation == "scientific":
        return format(value.normalize(), "E")
    if notation == "fixed":
        return format(value, "f")
    exponent = value.adjusted() if value != 0 else 0
    if value != 0 and (exponent >= 16 or exponent <= -7):
        return format(value.normalize(), "E")
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def calculate(expression: str, precision: int = 28, notation: str = "auto") -> dict[str, Any]:
    if not isinstance(expression, str) or not expression.strip():
        raise CalculatorError("expression 不能为空")
    if not isinstance(precision, int) or isinstance(precision, bool):
        raise CalculatorError("precision 必须是整数")
    if precision < 1 or precision > _MAX_PRECISION:
        raise CalculatorError(f"precision 必须在 1-{_MAX_PRECISION} 之间")
    if notation not in {"auto", "scientific", "fixed"}:
        raise CalculatorError("notation 只能是 auto/scientific/fixed")
    if len(expression) > 1000:
        raise CalculatorError("表达式过长")

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise CalculatorError(f"表达式语法错误: {exc.msg}") from exc

    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise CalculatorError(f"不允许的语法: {type(node).__name__}")
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise CalculatorError("只允许数值常量")

    def _eval(node: ast.AST) -> Decimal:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, float) and (
                node.value != node.value or node.value in (float("inf"), float("-inf"))
            ):
                raise CalculatorError("不允许的数值常量")
            return Decimal(str(node.value))
        if isinstance(node, ast.UnaryOp):
            value = _eval(node.operand)
            if isinstance(node.op, ast.UAdd):
                return +value
            if isinstance(node.op, ast.USub):
                return -value
            raise CalculatorError("不支持的一元运算符")
        if isinstance(node, ast.BinOp):
            left = _eval(node.left)
            right = _eval(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                result = left * right
                if len(result.as_tuple().digits) > _MAX_RESULT_DIGITS:
                    raise CalculatorError("结果位数过多")
                return result
            if isinstance(node.op, ast.Div):
                if right == 0:
                    raise CalculatorError("除数不能为 0")
                return left / right
            if isinstance(node.op, ast.Mod):
                if right == 0:
                    raise CalculatorError("取模除数不能为 0")
                return left % right
            if isinstance(node.op, ast.Pow):
                if right != right.to_integral_value():
                    raise CalculatorError("指数必须是整数")
                exponent = int(right)
                if abs(exponent) > _MAX_EXPONENT:
                    raise CalculatorError(f"指数绝对值不能超过 {_MAX_EXPONENT}")
                if abs(left) > 1 and exponent > 0 and abs(left).log10() * exponent > 5000:
                    raise CalculatorError("结果数值过大")
                return left**exponent
            raise CalculatorError("不支持的运算符")
        raise CalculatorError(f"不支持的表达式节点: {type(node).__name__}")

    try:
        with localcontext() as ctx:
            ctx.prec = precision
            result = _eval(tree)
            formatted = _format(result, precision, notation)
    except ZeroDivisionError as exc:
        raise CalculatorError("除数不能为 0") from exc
    except DivisionByZero as exc:
        raise CalculatorError("除数不能为 0") from exc
    except InvalidOperation as exc:
        raise CalculatorError(f"无效的数值运算: {exc}") from exc

    return {
        "expression": expression,
        "result": str(result),
        "formatted": formatted,
        "precision": precision,
        "notation": notation,
    }


__all__ = ["CalculatorError", "calculate"]

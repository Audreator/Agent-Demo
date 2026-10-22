from __future__ import annotations

from decimal import Decimal

import pytest

from app.mcp.tools.calculator import CalculatorError, calculate


def test_decimal_precision() -> None:
    result = calculate("0.1+0.2")
    assert result["result"] == "0.3"
    assert result["formatted"] == "0.3"


def test_scientific_notation_input() -> None:
    result = calculate("1.5e3 + 2.5e2")
    assert Decimal(result["result"]) == Decimal("1750")
    assert result["formatted"] == "1750"


def test_large_number_auto_scientific() -> None:
    result = calculate("1.23e20 * 2")
    assert "E" in result["formatted"].upper()


def test_precision_controls_division() -> None:
    result = calculate("1/3", precision=5)
    assert result["result"] == "0.33333"


def test_power_and_modulo() -> None:
    assert calculate("2**10")["result"] == "1024"
    assert calculate("10 % 3")["result"] == "1"


def test_division_by_zero() -> None:
    with pytest.raises(CalculatorError, match="除数不能为 0"):
        calculate("1/0")


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('dir')",
        "abs(1)",
        "(1).real",
        "'a' + 'b'",
        "True + 1",
        "[1, 2][0]",
    ],
)
def test_rejects_unsafe_syntax(expression: str) -> None:
    with pytest.raises(CalculatorError):
        calculate(expression)


def test_rejects_bad_notation_and_precision() -> None:
    with pytest.raises(CalculatorError):
        calculate("1+1", notation="binary")
    with pytest.raises(CalculatorError):
        calculate("1+1", precision=0)


def test_exponent_guard() -> None:
    with pytest.raises(CalculatorError):
        calculate("2**99999")
    with pytest.raises(CalculatorError):
        calculate("2**1.5")


def test_notation_options() -> None:
    assert calculate("1000", notation="scientific")["formatted"] == "1E+3"
    assert calculate("1.500", notation="fixed")["formatted"] == "1.5"

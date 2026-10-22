from __future__ import annotations

from app.context.manager import estimate_tokens
from app.errors import (
    BudgetExceededError,
    ErrorClass,
    StateError,
    ToolExecutionError,
    classify_http_error,
    classify_status,
)


def test_classify_status() -> None:
    assert classify_status(429) == ErrorClass.TRANSIENT
    assert classify_status(500) == ErrorClass.TRANSIENT
    assert classify_status(401) == ErrorClass.PERMANENT
    assert classify_status(403) == ErrorClass.PERMANENT
    assert classify_status(404) == ErrorClass.PERMANENT
    assert classify_status(400) == ErrorClass.PERMANENT


def test_classify_http_error_types() -> None:
    assert classify_http_error(429, "rate").error_class == ErrorClass.TRANSIENT
    assert classify_http_error(401, "auth").error_class == ErrorClass.PERMANENT


def test_estimate_tokens() -> None:
    assert estimate_tokens("a" * 160) == 100
    assert estimate_tokens("") == 1


def test_exception_classes() -> None:
    assert StateError("db down").error_class == ErrorClass.STATE
    assert ToolExecutionError("tool failed").error_class == ErrorClass.TOOL
    assert BudgetExceededError("over").error_class == ErrorClass.BUDGET

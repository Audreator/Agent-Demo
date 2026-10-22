from __future__ import annotations

from enum import StrEnum


class ErrorClass(StrEnum):
    TRANSIENT = "transient"
    RECOVERABLE = "recoverable"
    PERMANENT = "permanent"
    TOOL = "tool"
    BUDGET = "budget"
    STATE = "state"


class AgentError(Exception):
    error_class: ErrorClass = ErrorClass.RECOVERABLE
    code: str = "AGENT_ERROR"
    retryable: bool = False

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        status_code: int | None = None,
        data: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        if code:
            self.code = code
        self.status_code = status_code
        self.data = data or {}

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": self.message,
            "class": str(self.error_class),
            "status_code": self.status_code,
            "data": self.data,
        }


class TransientError(AgentError):
    error_class = ErrorClass.TRANSIENT
    code = "TRANSIENT"
    retryable = True


class RecoverableError(AgentError):
    error_class = ErrorClass.RECOVERABLE
    code = "RECOVERABLE"


class PermanentError(AgentError):
    error_class = ErrorClass.PERMANENT
    code = "PERMANENT"


class ToolExecutionError(AgentError):
    error_class = ErrorClass.TOOL
    code = "TOOL_EXECUTION"
    retryable = True


class BudgetExceededError(AgentError):
    error_class = ErrorClass.BUDGET
    code = "BUDGET_EXCEEDED"


class StateError(AgentError):
    error_class = ErrorClass.STATE
    code = "STATE_ERROR"


class ParseError(RecoverableError):
    code = "PARSE_ERROR"


class ServiceUnavailableError(TransientError):
    code = "MODEL_UNAVAILABLE"


class SessionNotFoundError(PermanentError):
    code = "SESSION_NOT_FOUND"


class TaskTurnLimitError(PermanentError):
    code = "TASK_TURN_LIMIT"


class MCPError(AgentError):
    code = "MCP_ERROR"

    def __init__(
        self,
        message: str,
        *,
        code: int | str | None = None,
        data: object = None,
    ) -> None:
        super().__init__(
            message,
            code=f"MCP_{code}" if code is not None else "MCP_ERROR",
            data={"mcp": data} if data is not None else {},
        )


class MCPTimeoutError(TransientError):
    code = "MCP_TIMEOUT"


class MCPProcessError(StateError):
    code = "MCP_PROCESS"


class MCPStartError(StateError):
    code = "MCP_START"


def classify_status(status_code: int) -> ErrorClass:
    if status_code in (401, 403, 404):
        return ErrorClass.PERMANENT
    if status_code == 429 or status_code >= 500:
        return ErrorClass.TRANSIENT
    if 400 <= status_code < 500:
        return ErrorClass.PERMANENT
    return ErrorClass.TRANSIENT


def classify_http_error(status_code: int, message: str) -> AgentError:
    kind = classify_status(status_code)
    if kind is ErrorClass.PERMANENT:
        return PermanentError(message, status_code=status_code)
    return TransientError(message, status_code=status_code)


__all__ = [
    "AgentError",
    "BudgetExceededError",
    "ErrorClass",
    "MCPError",
    "MCPProcessError",
    "MCPStartError",
    "MCPTimeoutError",
    "ParseError",
    "PermanentError",
    "RecoverableError",
    "ServiceUnavailableError",
    "SessionNotFoundError",
    "StateError",
    "TaskTurnLimitError",
    "ToolExecutionError",
    "TransientError",
    "classify_http_error",
    "classify_status",
]

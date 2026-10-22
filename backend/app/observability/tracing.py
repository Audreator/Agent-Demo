from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

from app.config import Settings, get_settings
from app.db import get_sessionmaker
from app.models.db_models import Event

_trace_id: ContextVar[str | None] = ContextVar("trace_id", default=None)
_span_stack: ContextVar[tuple[str, ...]] = ContextVar("span_stack", default=())
_session_id: ContextVar[str | None] = ContextVar("session_id", default=None)
_turn_no: ContextVar[int | None] = ContextVar("turn_no", default=None)

F = TypeVar("F", bound=Callable[..., Awaitable[Any]])


def new_id() -> str:
    return uuid.uuid4().hex


def current_trace_id() -> str:
    value = _trace_id.get()
    if value is None:
        value = new_id()
        _trace_id.set(value)
    return value


def set_trace_id(trace_id: str) -> None:
    _trace_id.set(trace_id)


def current_session_id() -> str | None:
    return _session_id.get()


def current_turn_no() -> int | None:
    return _turn_no.get()


def bind_context(session_id: str | None = None, turn_no: int | None = None) -> None:
    if session_id is not None:
        _session_id.set(session_id)
    if turn_no is not None:
        _turn_no.set(turn_no)


def reset_context() -> None:
    _trace_id.set(None)
    _span_stack.set(())
    _session_id.set(None)
    _turn_no.set(None)


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    return repr(value)


class DailyJsonFileHandler(logging.Handler):
    def __init__(self, log_dir: Path) -> None:
        super().__init__()
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._date: str | None = None
        self._stream: Any = None

    def _ensure_stream(self) -> Any:
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        if self._stream is None or self._date != today:
            if self._stream is not None:
                self._stream.close()
            self._date = today
            path = self.log_dir / f"app-{today}.jsonl"
            self._stream = path.open("a", encoding="utf-8")
        return self._stream

    def emit(self, record: logging.LogRecord) -> None:
        try:
            stream = self._ensure_stream()
            payload = getattr(record, "payload", None)
            entry = {
                "ts": datetime.now(UTC).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "trace_id": current_trace_id(),
                "session_id": current_session_id(),
                "turn_no": current_turn_no(),
                "payload": payload,
            }
            if record.exc_info:
                entry["exc_info"] = self.format(record)
            stream.write(json.dumps(entry, ensure_ascii=False, default=_json_default) + "\n")
            stream.flush()
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None
        super().close()


_logger: logging.Logger | None = None


def get_logger() -> logging.Logger:
    global _logger
    if _logger is not None:
        return _logger
    logger = logging.getLogger("agent")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        logger.addHandler(DailyJsonFileHandler(get_settings().log_dir))
    _logger = logger
    return logger


@dataclass
class SpanInfo:
    trace_id: str
    span_id: str
    parent_span_id: str | None
    kind: str
    started_at: float = field(default_factory=time.monotonic)
    fields: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def set(self, **values: Any) -> None:
        self.fields.update(values)


def _payload_preview(payload: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, str) and len(value) > 500:
            out[key] = value[:500] + f"...({len(value)} chars)"
        elif isinstance(value, dict):
            out[key] = _payload_preview(value)
        else:
            out[key] = value
    return out


class Tracer:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def new_trace(self) -> str:
        trace = new_id()
        set_trace_id(trace)
        _span_stack.set(())
        return trace

    async def emit_event(
        self,
        kind: str,
        payload: dict[str, Any] | None = None,
        *,
        trace_id: str | None = None,
        session_id: str | None = None,
        turn_no: int | None = None,
        duration_ms: int | None = None,
        span: SpanInfo | None = None,
    ) -> None:
        trace = trace_id or (span.trace_id if span else current_trace_id())
        payload = payload or {}
        event = {
            "trace_id": trace,
            "kind": kind,
            "payload": _payload_preview(payload),
            "duration_ms": duration_ms,
        }
        logger = get_logger()
        logger.info(kind, extra={"payload": event})
        try:
            sessionmaker = get_sessionmaker(self.settings)
            async with sessionmaker() as db:
                db.add(
                    Event(
                        trace_id=trace,
                        span_id=span.span_id if span else (span_id_for_event()),
                        parent_span_id=(span.parent_span_id if span else current_parent_span()),
                        session_id=session_id or current_session_id(),
                        turn_no=turn_no if turn_no is not None else current_turn_no(),
                        kind=kind,
                        payload=json.loads(
                            json.dumps(payload, ensure_ascii=False, default=_json_default)
                        ),
                        duration_ms=duration_ms,
                    )
                )
                await db.commit()
        except Exception as exc:  # observability must never break business flow
            logger.warning("event_persist_failed", extra={"payload": {"error": str(exc)}})

    @asynccontextmanager
    async def span(self, kind: str, **fields: Any) -> AsyncIterator[SpanInfo]:
        trace_id = current_trace_id()
        parent = _span_stack.get()
        parent_id = parent[-1] if parent else None
        span = SpanInfo(
            trace_id=trace_id,
            span_id=new_id()[:16],
            parent_span_id=parent_id,
            kind=kind,
            fields=fields,
        )
        _span_stack.set((*parent, span.span_id))
        try:
            yield span
        except Exception as exc:
            span.error = f"{type(exc).__name__}: {exc}"
            span.set(error_code=getattr(exc, "code", None))
            raise
        finally:
            _span_stack.set(parent)
            duration = int((time.monotonic() - span.started_at) * 1000)
            payload = dict(span.fields)
            if span.error:
                payload["error"] = span.error
            await self.emit_event(
                kind, payload, duration_ms=duration, trace_id=trace_id, span=span
            )


def span_id_for_event() -> str:
    stack = _span_stack.get()
    return stack[-1] if stack else new_id()[:16]


def current_parent_span() -> str | None:
    stack = _span_stack.get()
    return stack[-2] if len(stack) >= 2 else None


def traced(kind: str, **static_fields: Any) -> Callable[[F], F]:
    def decorator(fn: F) -> F:
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            async with get_tracer().span(kind, **static_fields):
                return await fn(*args, **kwargs)

        wrapper.__name__ = fn.__name__
        wrapper.__doc__ = fn.__doc__
        return wrapper  # type: ignore[return-value]

    return decorator


_tracer: Tracer | None = None


def get_tracer(settings: Settings | None = None) -> Tracer:
    global _tracer
    if _tracer is None or settings is not None:
        _tracer = Tracer(settings)
    return _tracer


def configure_logging(settings: Settings | None = None) -> None:
    settings = settings or get_settings()
    logger = get_logger()
    logger.handlers.clear()
    logger.addHandler(DailyJsonFileHandler(settings.log_dir))
    logger.setLevel(logging.INFO)


def configure_stderr_logging(level: int = logging.INFO) -> None:
    """MCP server logging: stderr only, stdout stays protocol-clean."""
    logger = logging.getLogger("agent.mcp")
    logger.setLevel(level)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    logging.getLogger("agent").addHandler(logging.StreamHandler(sys.stderr))


async def flush_events() -> None:
    await asyncio.sleep(0)


__all__ = [
    "Tracer",
    "bind_context",
    "configure_logging",
    "configure_stderr_logging",
    "current_parent_span",
    "current_session_id",
    "current_trace_id",
    "current_turn_no",
    "get_logger",
    "get_tracer",
    "new_id",
    "reset_context",
    "set_trace_id",
    "span_id_for_event",
    "traced",
]

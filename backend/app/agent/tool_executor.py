from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from app.config import Settings
from app.context.manager import tool_result_summary
from app.errors import MCPError, ToolExecutionError
from app.mcp.client import MCPClient
from app.mcp.tools.artifacts import save_artifact
from app.observability.tracing import Tracer


@dataclass
class ToolExecutionResult:
    name: str
    ok: bool
    content: str = ""
    data: Any = None
    summary: str = ""
    result_ref: str | None = None
    error: str | None = None
    latency_ms: int = 0
    attempts: int = 0
    arguments: dict[str, Any] = field(default_factory=dict)
    artifact_id: str | None = None


class ToolExecutor:
    def __init__(
        self,
        mcp: MCPClient,
        settings: Settings,
        *,
        tracer: Tracer,
        user_id: str,
    ) -> None:
        self.mcp = mcp
        self.settings = settings
        self.tracer = tracer
        self.user_id = user_id

    async def fetch_registry(self) -> dict[str, dict[str, Any]]:
        tools = await self.mcp.list_tools()
        return {tool["name"]: tool for tool in tools}

    def _prepare_arguments(
        self, name: str, arguments: dict[str, Any], *, session_id: UUID, turn_no: int
    ) -> dict[str, Any]:
        prepared = dict(arguments)
        if name == "todo" and prepared.get("action") in {"create", "update", "complete", "delete"}:
            payload = json.dumps(prepared, ensure_ascii=False, sort_keys=True)
            digest = hashlib.sha256(
                f"{session_id}:{turn_no}:{name}:{payload}".encode()
            ).hexdigest()[:24]
            prepared.setdefault("client_op_id", f"agent-{digest}")
        return prepared

    async def execute(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        session_id: UUID,
        turn_no: int,
        tool_call_id: str,
    ) -> ToolExecutionResult:
        prepared = self._prepare_arguments(
            name, arguments, session_id=session_id, turn_no=turn_no
        )
        started = time.monotonic()
        last_error: str | None = None
        attempts = 0
        for attempt in range(1, max(1, self.settings.tool_retry) + 1):
            attempts = attempt
            try:
                async with self.tracer.span(
                    "tool_call",
                    tool_name=name,
                    tool_call_id=tool_call_id,
                    attempt=attempt,
                    session_id=str(session_id),
                    turn_no=turn_no,
                    arguments_summary=_args_summary(prepared),
                ) as span:
                    await self.tracer.emit_event(
                        "mcp_request",
                        {
                            "method": "tools/call",
                            "tool_name": name,
                            "tool_call_id": tool_call_id,
                            "attempt": attempt,
                        },
                        session_id=str(session_id),
                        turn_no=turn_no,
                    )
                    response = await self.mcp.call_tool(
                        name,
                        prepared,
                        user_id=self.user_id,
                        session_id=str(session_id),
                        tool_call_id=tool_call_id,
                    )
                    text, data, is_error = _parse_tool_response(response)
                    latency = int((time.monotonic() - started) * 1000)
                    await self.tracer.emit_event(
                        "mcp_response",
                        {
                            "method": "tools/call",
                            "tool_name": name,
                            "tool_call_id": tool_call_id,
                            "is_error": is_error,
                            "result_size": len(text),
                            "latency_ms": latency,
                        },
                        session_id=str(session_id),
                        turn_no=turn_no,
                    )
                    span.set(result_size=len(text), is_error=is_error, latency_ms=latency)
                if is_error:
                    raise ToolExecutionError(text or f"工具 {name} 返回错误")
                result_ref, artifact_id = self._maybe_store_artifact(
                    tool_call_id, text, data, session_id
                )
                summary = _summarize(name, data, text)
                await self.tracer.emit_event(
                    "tool_result",
                    {
                        "tool_name": name,
                        "tool_call_id": tool_call_id,
                        "status": "ok",
                        "result_size": len(text),
                        "latency_ms": latency,
                        "result_ref": result_ref,
                    },
                    session_id=str(session_id),
                    turn_no=turn_no,
                )
                return ToolExecutionResult(
                    name=name,
                    ok=True,
                    content=text,
                    data=data,
                    summary=summary,
                    result_ref=result_ref,
                    latency_ms=latency,
                    attempts=attempt,
                    arguments=prepared,
                    artifact_id=artifact_id,
                )
            except Exception as exc:
                last_error = (
                    exc.message if hasattr(exc, "message") else f"{type(exc).__name__}: {exc}"
                )
                if isinstance(exc, MCPError) and exc.data:
                    last_error = f"{last_error} data={exc.data}"[:500]
                await self.tracer.emit_event(
                    "tool_attempt_failed",
                    {
                        "tool_name": name,
                        "tool_call_id": tool_call_id,
                        "attempt": attempt,
                        "error": last_error,
                    },
                    session_id=str(session_id),
                    turn_no=turn_no,
                )
                if isinstance(exc, MCPError) and str(exc.code).endswith("-32602"):
                    # 参数/Schema 错误属于可恢复输入问题，交给模型自纠，不重复调用工具。
                    break
                if attempt < max(1, self.settings.tool_retry):
                    await asyncio.sleep(0.15 * attempt)
        latency = int((time.monotonic() - started) * 1000)
        await self.tracer.emit_event(
            "tool_result",
            {
                "tool_name": name,
                "tool_call_id": tool_call_id,
                "status": "error",
                "error": last_error,
                "latency_ms": latency,
            },
            session_id=str(session_id),
            turn_no=turn_no,
        )
        return ToolExecutionResult(
            name=name,
            ok=False,
            summary=last_error or "工具执行失败",
            error=last_error or "工具执行失败",
            latency_ms=latency,
            attempts=attempts,
            arguments=prepared,
        )

    def _maybe_store_artifact(
        self, tool_call_id: str, text: str, data: Any, session_id: UUID
    ) -> tuple[str | None, str | None]:
        if isinstance(data, dict) and isinstance(data.get("artifact"), str):
            ref = data["artifact"]
            return ref, ref.removeprefix("artifact://")
        if len(text) > self.settings.tool_result_context_limit:
            ref = save_artifact(
                text,
                artifact_id=tool_call_id,
                session_id=str(session_id),
                base_dir=self.settings.artifacts_dir,
            )
            return ref, tool_call_id
        return None, None


def _parse_tool_response(response: dict[str, Any]) -> tuple[str, Any, bool]:
    content = response.get("content") or []
    texts: list[str] = []
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                texts.append(str(item.get("text", "")))
            elif isinstance(item, str):
                texts.append(item)
    text = "\n".join(texts).strip()
    data: Any = None
    try:
        data = json.loads(text) if text else None
    except json.JSONDecodeError:
        data = None
    is_error = bool(response.get("isError"))
    return text, data, is_error


def _summarize(name: str, data: Any, text: str) -> str:
    if isinstance(data, dict):
        if name == "calculator":
            return f"{data.get('expression')} = {data.get('formatted') or data.get('result')}"
        if name == "search":
            results = data.get("results") or []
            titles = "；".join(
                str(item.get("title")) for item in results[:3] if isinstance(item, dict)
            )
            return f"搜索到 {len(results)} 条结果：{titles}"[:500]
        if name == "todo":
            todo = data.get("todo")
            if isinstance(todo, dict):
                return f"待办 {todo.get('id')}: {todo.get('content')} ({todo.get('status')})"[:500]
            return f"待办列表共 {data.get('count', 0)} 条"[:500]
        if name == "read_docs":
            return (
                f"读取 {data.get('source')} 偏移 {data.get('offset')}，"
                f"返回 {data.get('returned_chars')} 字符"
            )[:500]
        if name == "weather":
            return str(data.get("summary") or "")[:500]
        return tool_result_summary(data)
    return text[:500]


def _args_summary(arguments: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key, value in arguments.items():
        if key in {"client_op_id", "query", "expression", "content", "path", "ref"}:
            text = str(value)
            summary[key] = text[:120] + ("…" if len(text) > 120 else "")
        elif isinstance(value, (int, float, bool)) or value is None:
            summary[key] = value
        else:
            summary[key] = str(value)[:80]
    return summary


__all__ = ["ToolExecutionResult", "ToolExecutor"]

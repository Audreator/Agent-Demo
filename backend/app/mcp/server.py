from __future__ import annotations

import asyncio
import json
import logging
import sys
from collections.abc import Awaitable, Callable
from typing import Any

from app.mcp.schema import SchemaValidationError, validate_arguments
from app.mcp.tools.calculator import CalculatorError, calculate
from app.mcp.tools.read_docs import DocsError, read_document
from app.mcp.tools.search import search_tool
from app.mcp.tools.todo import TodoError, todo_tool
from app.mcp.tools.weather import WeatherError, get_weather
from app.observability import configure_stderr_logging

PROTOCOL_VERSION = "2024-11-05"
TOOLS: dict[str, dict[str, Any]] = {}
logger = logging.getLogger("agent.mcp.server")

ToolFunc = Callable[..., Awaitable[Any]]


def register(
    name: str, description: str, input_schema: dict[str, Any]
) -> Callable[[ToolFunc], ToolFunc]:
    def decorator(fn: ToolFunc) -> ToolFunc:
        TOOLS[name] = {
            "name": name,
            "description": description,
            "inputSchema": input_schema,
            "_fn": fn,
        }
        return fn

    return decorator


@register(
    "calculator",
    "精确数值计算器。支持加减乘除、取模、乘方、括号和科学计数法，使用 Decimal 保证精度。",
    {
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "要计算的数学表达式"},
            "precision": {"type": "integer", "default": 28, "minimum": 1, "maximum": 100},
            "notation": {
                "type": "string",
                "enum": ["auto", "scientific", "fixed"],
                "default": "auto",
            },
        },
        "required": ["expression"],
        "additionalProperties": False,
    },
)
async def _calculator(
    expression: str, precision: int = 28, notation: str = "auto", _meta: Any = None
) -> dict[str, Any]:
    return calculate(expression, precision=precision, notation=notation)


@register(
    "todo",
    "待办事项管理。支持 create/list/update/complete/delete，"
    "数据按用户归属持久化到 PostgreSQL。写操作携带 client_op_id 保证幂等。",
    {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["create", "list", "update", "complete", "delete"],
            },
            "id": {"type": "string", "description": "待办 UUID，更新/完成/删除时必填"},
            "content": {"type": "string", "description": "待办内容"},
            "due_at": {"type": "string", "description": "ISO 8601 截止时间"},
            "status": {"type": "string", "enum": ["open", "done"]},
            "client_op_id": {"type": "string", "description": "客户端操作号，用于幂等"},
        },
        "required": ["action"],
        "additionalProperties": False,
    },
)
async def _todo(**kwargs: Any) -> dict[str, Any]:
    meta = kwargs.pop("_meta", {}) or {}
    return await todo_tool(
        action=str(kwargs.get("action") or ""),
        user_id=meta.get("user_id"),
        session_id=meta.get("session_id"),
        id=kwargs.get("id"),
        content=kwargs.get("content"),
        due_at=kwargs.get("due_at"),
        status=kwargs.get("status"),
        client_op_id=kwargs.get("client_op_id"),
    )


@register(
    "search",
    "联网搜索。固定使用 DeepSeek 原生联网搜索，返回标题、链接和摘要；"
    "结果超长会截断并写入 artifact。",
    {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "max_results": {"type": "integer", "default": 5, "minimum": 1, "maximum": 10},
            "recency_days": {"type": "integer", "minimum": 1, "maximum": 3650},
            "site": {"type": "string", "maxLength": 200},
        },
        "required": ["query"],
        "additionalProperties": False,
    },
)
async def _search(**kwargs: Any) -> dict[str, Any]:
    meta = kwargs.pop("_meta", {}) or {}
    return await search_tool(
        query=kwargs.get("query", ""),
        max_results=int(kwargs.get("max_results", 5)),
        recency_days=kwargs.get("recency_days"),
        site=kwargs.get("site"),
        artifact_id=meta.get("tool_call_id"),
        session_id=meta.get("session_id"),
    )


@register(
    "weather",
    "天气查询。通过 Open-Meteo 免 key 接口做地名解析和多天预报，返回中文天气描述。",
    {
        "type": "object",
        "properties": {
            "location": {"type": "string"},
            "days": {"type": "integer", "default": 1, "minimum": 1, "maximum": 7},
            "units": {"type": "string", "enum": ["metric", "imperial"], "default": "metric"},
        },
        "required": ["location"],
        "additionalProperties": False,
    },
)
async def _weather(**kwargs: Any) -> dict[str, Any]:
    kwargs.pop("_meta", None)
    return await get_weather(
        location=kwargs.get("location", ""),
        days=int(kwargs.get("days", 1)),
        units=kwargs.get("units", "metric"),
    )


@register(
    "read_docs",
    "读取文档。path 限定在 DOCS_ROOT 内，支持 offset/limit 分段；"
    "也可用 ref=artifact://{id} 回读大工具结果。",
    {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "ref": {"type": "string"},
            "offset": {"type": "integer", "default": 0, "minimum": 0},
            "limit": {"type": "integer", "default": 4000, "minimum": 1, "maximum": 100000},
        },
        "additionalProperties": False,
    },
)
async def _read_docs(**kwargs: Any) -> dict[str, Any]:
    meta = kwargs.pop("_meta", {}) or {}
    return read_document(
        path=kwargs.get("path"),
        ref=kwargs.get("ref"),
        offset=int(kwargs.get("offset", 0)),
        limit=int(kwargs.get("limit", 4000)),
        session_id=meta.get("session_id"),
    )


def ok(request_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def err(request_id: Any, code: int, message: str, data: Any = None) -> dict[str, Any]:
    error: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    return {"jsonrpc": "2.0", "id": request_id, "error": error}


async def dispatch(message: dict[str, Any]) -> dict[str, Any] | None:
    method = message.get("method")
    request_id = message.get("id")
    params = message.get("params") or {}
    if not isinstance(params, dict):
        return err(request_id, -32602, "params 必须是对象")

    if method == "initialize":
        return ok(
            request_id,
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "agent-tools", "version": "0.1"},
            },
        )
    if method and str(method).startswith("notifications/"):
        return None
    if method == "ping":
        return ok(request_id, {})
    if method == "tools/list":
        return ok(
            request_id,
            {
                "tools": [
                    {
                        "name": tool["name"],
                        "description": tool["description"],
                        "inputSchema": tool["inputSchema"],
                    }
                    for tool in TOOLS.values()
                ]
            },
        )
    if method == "tools/call":
        name = params.get("name")
        arguments = params.get("arguments") or {}
        meta = params.get("_meta") or {}
        if not isinstance(name, str) or name not in TOOLS:
            return err(request_id, -32602, f"未知工具: {name}")
        if not isinstance(arguments, dict):
            return err(request_id, -32602, "参数必须是 JSON 对象")
        try:
            validate_arguments(TOOLS[name]["inputSchema"], arguments)
        except SchemaValidationError as exc:
            return err(request_id, -32602, f"参数不符合 Schema: {exc}", {"tool": name})
        call_args = dict(arguments)
        call_args["_meta"] = meta
        try:
            result = await TOOLS[name]["_fn"](**call_args)
        except (
            CalculatorError,
            TodoError,
            DocsError,
            WeatherError,
            ValueError,
        ) as exc:
            return ok(
                request_id,
                {
                    "content": [{"type": "text", "text": f"{type(exc).__name__}: {exc}"}],
                    "isError": True,
                },
            )
        except Exception as exc:
            logger.exception("tool execution failed: %s", name)
            return ok(
                request_id,
                {
                    "content": [{"type": "text", "text": f"{type(exc).__name__}: {exc}"}],
                    "isError": True,
                },
            )
        return ok(
            request_id,
            {
                "content": [
                    {"type": "text", "text": json.dumps(result, ensure_ascii=False, default=str)}
                ],
                "isError": False,
            },
        )
    return err(request_id, -32601, f"method not found: {method}")


class GuardedStdout:
    """Only JSON-RPC lines may pass through stdout; anything else is redirected to stderr."""

    def __init__(self, stream: Any) -> None:
        self._stream = stream

    def write(self, text: str) -> int:
        if not text:
            return 0
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                parsed = None
            if (
                not isinstance(parsed, dict)
                or parsed.get("jsonrpc") != "2.0"
                or not any(key in parsed for key in ("method", "result", "error"))
            ):
                sys.stderr.write(
                    f"[stdout-guard] discarded non-protocol output: {stripped[:300]}\n"
                )
                sys.stderr.flush()
                continue
            self._stream.write(line + "\n")
        return len(text)

    def flush(self) -> None:
        self._stream.flush()

    def isatty(self) -> bool:
        return False


def install_stdout_guard() -> GuardedStdout:
    guarded = GuardedStdout(sys.stdout)
    sys.stdout = guarded  # type: ignore[assignment]
    return guarded


def stdout_purity_check() -> dict[str, Any]:
    root = logging.getLogger()
    streams = []
    for handler in root.handlers + logging.getLogger("agent.mcp").handlers:
        stream = getattr(handler, "stream", None)
        if stream is not None:
            streams.append(stream)
    polluted = any(stream is sys.__stdout__ for stream in streams)
    guarded = isinstance(sys.stdout, GuardedStdout)
    if polluted:
        sys.stderr.write("stdout purity check failed: logging handler writes to stdout\n")
    if not guarded:
        sys.stderr.write("stdout purity check failed: stdout guard not installed\n")
    return {"pure": guarded and not polluted, "guarded": guarded, "logging_to_stdout": polluted}


async def serve(stdin: Any = None, stdout: Any = None) -> None:
    input_stream = stdin or sys.stdin
    output_stream = stdout or sys.stdout
    while True:
        line = await asyncio.to_thread(input_stream.readline)
        if not line:
            break
        text = line.strip()
        if not text:
            continue
        response: dict[str, Any] | None = None
        try:
            message = json.loads(text)
        except json.JSONDecodeError:
            response = err(None, -32700, "Parse error")
            output_stream.write(json.dumps(response, ensure_ascii=False) + "\n")
            output_stream.flush()
            continue
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
            request_id = message.get("id") if isinstance(message, dict) else None
            response = err(request_id, -32600, "Invalid Request")
        else:
            response = await dispatch(message)
        if response is not None:
            output_stream.write(json.dumps(response, ensure_ascii=False) + "\n")
            output_stream.flush()


def main() -> None:
    configure_stderr_logging()
    guard = install_stdout_guard()
    check = stdout_purity_check()
    logger.info("MCP server starting, stdout purity=%s", check)
    sys.stderr.write(f"MCP server stdout purity check: {check}\n")
    sys.stderr.flush()
    try:
        asyncio.run(serve(stdout=guard))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

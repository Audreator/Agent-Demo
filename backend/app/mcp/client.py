from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from app.errors import MCPError, MCPProcessError, MCPStartError, MCPTimeoutError

logger = logging.getLogger("agent.mcp.client")

PROTOCOL_VERSION = "2024-11-05"


class MCPClient:
    """Hand-written JSON-RPC 2.0 client over newline-delimited stdio."""

    def __init__(
        self,
        cmd: list[str],
        *,
        env: dict[str, str] | None = None,
        cwd: str | Path | None = None,
        timeout: float = 30.0,
        name: str = "agent-client",
    ) -> None:
        self._cmd = cmd
        self._env = dict(os.environ if env is None else env)
        self._cwd = str(cwd) if cwd is not None else None
        self._timeout = timeout
        self._name = name
        self._proc: asyncio.subprocess.Process | None = None
        self._next_id = 0
        self._pending: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self._reader_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._write_lock = asyncio.Lock()
        self._started = False
        self._tools: list[dict[str, Any]] = []
        self.notifications: list[dict[str, Any]] = []
        self.on_notification: Callable[[dict[str, Any]], None] | None = None

    @property
    def started(self) -> bool:
        return self._started and self._proc is not None and self._proc.returncode is None

    @property
    def process_id(self) -> int | None:
        return self._proc.pid if self._proc else None

    async def start(self) -> None:
        if self.started:
            return
        try:
            self._proc = await asyncio.create_subprocess_exec(
                *self._cmd,
                cwd=self._cwd,
                env=self._env,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=4 * 1024 * 1024,
            )
        except (OSError, ValueError) as exc:
            raise MCPStartError(f"MCP 子进程启动失败: {exc}") from exc
        self._reader_task = asyncio.create_task(self._read_loop(), name="mcp-reader")
        self._stderr_task = asyncio.create_task(self._stderr_loop(), name="mcp-stderr")
        try:
            result = await self.call(
                "initialize",
                {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": self._name, "version": "0.1"},
                },
            )
            await self.notify("notifications/initialized", {})
        except Exception:
            await self.close()
            raise
        server = result.get("serverInfo") if isinstance(result, dict) else None
        logger.info("MCP initialized: %s", server)
        self._started = True

    async def _stderr_loop(self) -> None:
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        while True:
            line = await proc.stderr.readline()
            if not line:
                break
            text = line.decode("utf-8", errors="replace").rstrip()
            if text:
                sys.stderr.write(f"[mcp-server] {text}\n")
        sys.stderr.flush()

    async def _read_loop(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        while True:
            try:
                line = await proc.stdout.readline()
            except ValueError as exc:
                self._fail_pending(
                    MCPProcessError(f"MCP stdout 单行超长或编码异常: {exc}")
                )
                break
            if not line:
                break
            text = line.decode("utf-8", errors="replace").strip()
            if not text:
                continue
            try:
                message = json.loads(text)
            except json.JSONDecodeError:
                logger.warning("Discarded non-JSON line from MCP stdout: %s", text[:200])
                continue
            if not isinstance(message, dict):
                continue
            msg_id = message.get("id")
            if "id" in message and msg_id in self._pending:
                future = self._pending.pop(msg_id)
                if not future.done():
                    future.set_result(message)
            else:
                self.notifications.append(message)
                if self.on_notification is not None:
                    try:
                        self.on_notification(message)
                    except Exception:
                        logger.exception("notification handler failed")
        self._fail_pending(MCPProcessError("MCP 子进程输出已关闭"))

    def _fail_pending(self, error: Exception) -> None:
        for future in list(self._pending.values()):
            if not future.done():
                future.set_exception(error)
        self._pending.clear()

    async def _write(self, payload: dict[str, Any]) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None:
            raise MCPProcessError("MCP 子进程未启动")
        if proc.returncode is not None:
            raise MCPProcessError(f"MCP 子进程已退出，returncode={proc.returncode}")
        data = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
        async with self._write_lock:
            try:
                proc.stdin.write(data)
                await proc.stdin.drain()
            except (BrokenPipeError, ConnectionResetError) as exc:
                raise MCPProcessError(f"MCP 管道写入失败: {exc}") from exc

    async def call(
        self, method: str, params: dict[str, Any] | None = None, timeout: float | None = None
    ) -> dict[str, Any]:
        self._next_id += 1
        request_id = self._next_id
        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        payload: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id, "method": method}
        if params is not None:
            payload["params"] = params
        try:
            await self._write(payload)
            message = await asyncio.wait_for(future, timeout=timeout or self._timeout)
        except TimeoutError as exc:
            raise MCPTimeoutError(f"MCP 请求超时: {method}") from exc
        finally:
            self._pending.pop(request_id, None)
        if "error" in message:
            error = message["error"] or {}
            raise MCPError(
                str(error.get("message", "MCP error")),
                code=error.get("code"),
                data=error.get("data"),
            )
        result = message.get("result")
        return result if isinstance(result, dict) else {"value": result}

    async def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        payload: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            payload["params"] = params
        await self._write(payload)

    async def list_tools(self) -> list[dict[str, Any]]:
        result = await self.call("tools/list", {})
        tools = result.get("tools", [])
        self._tools = list(tools)
        return self._tools

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        user_id: str | None = None,
        session_id: str | None = None,
        tool_call_id: str | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"name": name, "arguments": arguments}
        meta: dict[str, Any] = {}
        if user_id:
            meta["user_id"] = user_id
        if session_id:
            meta["session_id"] = session_id
        if tool_call_id:
            meta["tool_call_id"] = tool_call_id
        if meta:
            params["_meta"] = meta
        return await self.call("tools/call", params, timeout=timeout)

    async def close(self) -> None:
        proc = self._proc
        self._started = False
        if proc is not None and proc.returncode is None:
            try:
                if proc.stdin is not None and not proc.stdin.is_closing():
                    proc.stdin.close()
                proc.terminate()
                await asyncio.wait_for(proc.wait(), timeout=3)
            except (TimeoutError, ProcessLookupError):
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
        for task in (self._reader_task, self._stderr_task):
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
        self._reader_task = None
        self._stderr_task = None
        self._proc = None
        self._fail_pending(MCPProcessError("MCP 客户端已关闭"))


__all__ = ["MCPClient", "PROTOCOL_VERSION"]

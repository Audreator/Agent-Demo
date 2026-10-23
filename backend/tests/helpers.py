from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from app.mcp.server import TOOLS
from app.models.providers import ChatChunk, ProviderConfig


def registry() -> dict[str, dict[str, Any]]:
    return {
        name: {
            "name": tool["name"],
            "description": tool["description"],
            "inputSchema": tool["inputSchema"],
        }
        for name, tool in TOOLS.items()
    }


def tool_list() -> list[dict[str, Any]]:
    return list(registry().values())


@dataclass
class FakeMCP:
    started: bool = True
    process_id: int | None = 999
    tools: list[dict[str, Any]] = field(default_factory=tool_list)
    responses: dict[str, dict[str, Any]] = field(default_factory=dict)
    fail_tools: set[str] = field(default_factory=set)
    calls: list[tuple[str, dict[str, Any], dict[str, Any]]] = field(default_factory=list)

    async def list_tools(self) -> list[dict[str, Any]]:
        return self.tools

    async def start(self) -> None:
        self.started = True

    async def close(self) -> None:
        self.started = False

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
        self.calls.append((name, arguments, {"user_id": user_id, "session_id": session_id}))
        if name in self.fail_tools:
            return {
                "content": [{"type": "text", "text": "ToolFailure: upstream unavailable"}],
                "isError": True,
            }
        payload = self.responses.get(name, {"ok": True})
        return {
            "content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}],
            "isError": False,
        }


@dataclass
class ScriptedProvider:
    scripts: list[Any]
    name: str = "fake-l1"
    layer: str = "deepseek"
    calls: list[list[dict[str, Any]]] = field(default_factory=list)
    model: str = "fake-model"

    def __post_init__(self) -> None:
        self.config = ProviderConfig(
            layer=self.layer,
            name=self.name,
            base_url="http://fake.local/v1",
            api_key="test-key",
            model=self.model,
        )

    async def stream(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> AsyncIterator[ChatChunk]:
        self.calls.append(messages)
        if not self.scripts:
            yield ChatChunk(content="Thought: 无脚本\nFinal Answer: 脚本耗尽")
            return
        script = self.scripts.pop(0)
        if isinstance(script, Exception):
            raise script
        if isinstance(script, str):
            yield ChatChunk(content=script)
            return
        if isinstance(script, dict):
            from app.models.providers import merge_tool_call_deltas

            accumulated: dict[int, dict[str, Any]] = {}
            if script.get("text"):
                yield ChatChunk(content=str(script["text"]))
            if script.get("reasoning"):
                yield ChatChunk(reasoning=str(script["reasoning"]))
            if script.get("tool_calls"):
                merge_tool_call_deltas(accumulated, script["tool_calls"])
                yield ChatChunk(tool_calls=script["tool_calls"])
            if script.get("usage"):
                yield ChatChunk(usage=script["usage"])
            return
        for item in script:
            yield item


@dataclass
class EchoProvider:
    calls: list[list[dict[str, Any]]] = field(default_factory=list)
    name: str = "echo"
    layer: str = "deepseek"
    prefix: str = "回答"

    def __post_init__(self) -> None:
        self.config = ProviderConfig(
            layer=self.layer,
            name=self.name,
            base_url="http://fake.local/v1",
            api_key="test-key",
            model="echo-model",
        )

    async def stream(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> AsyncIterator[ChatChunk]:
        self.calls.append(messages)
        user_text = ""
        for message in reversed(messages):
            if message.get("role") == "user":
                user_text = str(message.get("content", ""))
                break
        text = f"Thought: 收到输入\nFinal Answer: {self.prefix}:{user_text[:40]}"
        yield ChatChunk(content=text)


@dataclass
class Collector:
    events: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    async def __call__(self, event: str, data: dict[str, Any]) -> None:
        self.events.append((event, data))

    def names(self) -> list[str]:
        return [name for name, _ in self.events]

    def first(self, name: str) -> dict[str, Any] | None:
        for event, data in self.events:
            if event == name:
                return data
        return None

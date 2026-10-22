from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import httpx

from app.errors import PermanentError, TransientError


@dataclass
class ProviderConfig:
    layer: str
    name: str
    base_url: str
    api_key: str
    model: str
    timeout: float = 60.0

    @property
    def configured(self) -> bool:
        if not self.base_url.strip():
            return False
        if self.layer != "local" and not self.api_key.strip():
            return False
        return True

    def snapshot(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "name": self.name,
            "base_url": self.base_url,
            "model": self.model,
            "configured": self.configured,
        }


@dataclass
class ChatChunk:
    content: str = ""
    reasoning: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, Any] | None = None
    finish_reason: str | None = None
    raw: dict[str, Any] | None = None


@dataclass
class ModelResult:
    text: str
    reasoning: str
    tool_calls: list[dict[str, Any]]
    usage: dict[str, Any]
    provider: str
    layer: str
    model: str

    def native_response(self) -> dict[str, Any]:
        message: dict[str, Any] = {"role": "assistant", "content": self.text}
        if self.reasoning:
            message["reasoning_content"] = self.reasoning
        if self.tool_calls:
            message["tool_calls"] = self.tool_calls
        return {"choices": [{"index": 0, "message": message, "finish_reason": "stop"}]}


class ChatProvider:
    def __init__(self, config: ProviderConfig) -> None:
        self.config = config

    @property
    def name(self) -> str:
        return self.config.name

    async def stream(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> AsyncIterator[ChatChunk]:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "stream": True,
            "temperature": temperature,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if max_tokens:
            payload["max_tokens"] = max_tokens
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        url = self.config.base_url.rstrip("/") + "/chat/completions"
        timeout = httpx.Timeout(self.config.timeout, connect=10.0)
        try:
            terminal_seen = False
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream("POST", url, json=payload, headers=headers) as response:
                    if response.status_code >= 400:
                        body = (await response.aread()).decode("utf-8", errors="replace")
                        retry_after = response.headers.get("Retry-After")
                        message = f"{self.name} 返回 {response.status_code}: {body[:300]}"
                        error_cls = (
                            TransientError
                            if response.status_code == 429 or response.status_code >= 500
                            else PermanentError
                        )
                        error = error_cls(message, status_code=response.status_code)
                        if retry_after:
                            error.data["retry_after"] = retry_after
                        raise error
                    async for line in response.aiter_lines():
                        chunk = _parse_sse_line(line)
                        if chunk is None:
                            continue
                        if chunk.get("_done"):
                            terminal_seen = True
                            break
                        if chunk.get("error"):
                            raise TransientError(
                                f"{self.name} 流内错误: {chunk.get('error')}"
                            )
                        converted = _to_chunk(chunk)
                        if converted.finish_reason:
                            terminal_seen = True
                        yield converted
                    if not terminal_seen:
                        raise TransientError(f"{self.name} 流式响应未正常结束")
        except httpx.TimeoutException as exc:
            raise TransientError(f"{self.name} 请求超时: {exc}") from exc
        except httpx.HTTPError as exc:
            raise TransientError(f"{self.name} 连接失败: {exc}") from exc


def _parse_sse_line(line: str) -> dict[str, Any] | None:
    text = line.strip()
    if not text or text.startswith(":"):
        return None
    if text.startswith("data:"):
        text = text[len("data:") :].strip()
    if text == "[DONE]":
        return {"_done": True}
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _to_chunk(data: dict[str, Any]) -> ChatChunk:
    choices = data.get("choices") or []
    if not choices:
        return ChatChunk(usage=data.get("usage"), raw=data)
    choice = choices[0] or {}
    delta = choice.get("delta") or choice.get("message") or {}
    tool_calls = delta.get("tool_calls") or []
    return ChatChunk(
        content=str(delta.get("content") or ""),
        reasoning=str(delta.get("reasoning_content") or delta.get("reasoning") or ""),
        tool_calls=list(tool_calls),
        usage=data.get("usage"),
        finish_reason=choice.get("finish_reason"),
        raw=data,
    )


def merge_tool_call_deltas(
    accumulated: dict[int, dict[str, Any]], deltas: list[dict[str, Any]]
) -> None:
    for delta in deltas:
        index = int(delta.get("index", 0))
        current = accumulated.setdefault(
            index,
            {
                "id": delta.get("id") or f"call_{index}",
                "type": "function",
                "function": {"name": "", "arguments": ""},
            },
        )
        if delta.get("id"):
            current["id"] = delta["id"]
        function = delta.get("function") or {}
        if function.get("name"):
            current["function"]["name"] += str(function["name"])
        if function.get("arguments"):
            arguments = function["arguments"]
            if isinstance(arguments, str):
                current["function"]["arguments"] += arguments
            else:
                current["function"]["arguments"] = json.dumps(arguments, ensure_ascii=False)


__all__ = [
    "ChatChunk",
    "ChatProvider",
    "ModelResult",
    "ProviderConfig",
    "merge_tool_call_deltas",
]

from __future__ import annotations

import json
from typing import Any

from app.agent.react_parser import ParsedOutput, repair_json, validate_tool_call


def normalize_native(
    response: dict[str, Any], registry: dict[str, dict[str, Any]]
) -> ParsedOutput:
    try:
        choice = response["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return ParsedOutput(kind="invalid", reason="模型响应缺少 choices[0].message")
    reasoning = str(choice.get("reasoning_content") or choice.get("reasoning") or "")
    tool_calls = choice.get("tool_calls") or []
    content = choice.get("content") or ""
    if not tool_calls:
        if not str(content).strip():
            return ParsedOutput(
                kind="invalid",
                thought=reasoning,
                reason="模型既未返回内容也未返回工具调用",
            )
        return ParsedOutput(kind="final", thought=reasoning, answer=str(content))
    if len(tool_calls) > 1:
        return ParsedOutput(
            kind="invalid",
            thought=reasoning,
            reason="一次只能调用一个工具，请逐步调用",
        )
    call = tool_calls[0]
    function = call.get("function") or {}
    name = str(function.get("name") or "").strip()
    raw_arguments = function.get("arguments") or "{}"
    if isinstance(raw_arguments, dict):
        arguments = raw_arguments
    else:
        arguments = repair_json(str(raw_arguments)) or {}
    if not arguments and str(raw_arguments).strip() not in ("{}", ""):
        return ParsedOutput(
            kind="invalid",
            thought=reasoning,
            reason="工具调用参数不是合法 JSON",
            call_id=call.get("id"),
        )
    problem = validate_tool_call(name, arguments, registry)
    if problem:
        return ParsedOutput(
            kind="invalid", thought=reasoning, reason=problem, call_id=call.get("id")
        )
    return ParsedOutput(
        kind="action",
        thought=reasoning,
        name=name,
        arguments=arguments,
        call_id=call.get("id"),
    )


def tools_as_openai(registry: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    tools = []
    for tool in registry.values():
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("inputSchema", {"type": "object", "properties": {}}),
                },
            }
        )
    return tools


def dump_arguments(arguments: dict[str, Any]) -> str:
    return json.dumps(arguments, ensure_ascii=False)


__all__ = ["dump_arguments", "normalize_native", "tools_as_openai"]

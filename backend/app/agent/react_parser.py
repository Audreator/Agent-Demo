from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from app.mcp.schema import SchemaValidationError, validate_arguments

_FIELD_PATTERN = re.compile(
    r"(?im)(?<![A-Za-z])(Thought|Action\s+Input|Action|Observation|Final\s+Answer)\s*[:：]"
)
_FENCE_PATTERN = re.compile(r"^\s*```[A-Za-z]*\s*$", re.M)


@dataclass
class ParsedOutput:
    kind: Literal["final", "action", "invalid"]
    thought: str = ""
    answer: str = ""
    name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    call_id: str | None = None
    reason: str = ""
    raw: str = ""


def strip_markdown_fences(text: str) -> str:
    lines = [line for line in text.splitlines() if not _FENCE_PATTERN.match(line)]
    return "\n".join(lines)


def split_fields(text: str) -> dict[str, str]:
    """Split ReAct fields, tolerating Chinese colons, noise and multi-line values."""
    cleaned = strip_markdown_fences(text)
    matches = list(_FIELD_PATTERN.finditer(cleaned))
    if not matches:
        return {}
    parts: dict[str, str] = {}
    for index, match in enumerate(matches):
        key = re.sub(r"\s+", " ", match.group(1)).lower()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(cleaned)
        value = cleaned[start:end].strip()
        if key in parts and parts[key]:
            parts[key] = parts[key] + "\n" + value
        else:
            parts[key] = value
    return parts


def _balanced_object(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return text[start:]


def repair_json(raw: str) -> dict[str, Any] | None:
    if raw is None:
        return None
    text = strip_markdown_fences(str(raw)).strip()
    text = text.replace("“", '"').replace("”", '"')
    if not text:
        return None
    candidates: list[str] = []
    candidates.append(text)
    block = _balanced_object(text)
    if block:
        candidates.append(block)
    for candidate in candidates:
        for attempt in _json_variants(candidate):
            try:
                value = json.loads(attempt)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
            if isinstance(value, list) and len(value) == 1 and isinstance(value[0], dict):
                return value[0]
    return None


def _json_variants(text: str) -> list[str]:
    variants = [text.strip()]
    cleaned = re.sub(r",\s*([\]}])", r"\1", text.strip())
    variants.append(cleaned)
    if cleaned.count("{") > cleaned.count("}"):
        variants.append(cleaned + "}" * (cleaned.count("{") - cleaned.count("}")))
    if cleaned.count("[") > cleaned.count("]"):
        variants.append(cleaned + "]" * (cleaned.count("[") - cleaned.count("]")))
    # Strip common prefix/suffix explanations around the JSON object.
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end > start:
        variants.append(cleaned[start : end + 1])
    return list(dict.fromkeys(variants))


def validate_tool_call(
    name: str, arguments: dict[str, Any], registry: dict[str, dict[str, Any]]
) -> str | None:
    if name not in registry:
        return f"未知工具: {name}，可用工具: {', '.join(sorted(registry))}"
    schema = registry[name].get("inputSchema") or {}
    try:
        validate_arguments(schema, arguments)
    except SchemaValidationError as exc:
        return f"工具 {name} 参数不符合 Schema: {exc}"
    return None


def parse_react(text: str, registry: dict[str, dict[str, Any]]) -> ParsedOutput:
    fields = split_fields(text)
    thought = fields.get("thought", "").strip()
    if "final answer" in fields and fields["final answer"].strip():
        return ParsedOutput(
            kind="final", thought=thought, answer=fields["final answer"].strip(), raw=text
        )
    if "action" in fields and fields["action"].strip():
        name = fields["action"].strip().strip("`'\"*").strip()
        name = name.splitlines()[0].strip()
        arguments = repair_json(fields.get("action input", ""))
        if arguments is None:
            return ParsedOutput(
                kind="invalid",
                thought=thought,
                reason="Action Input 不是合法 JSON，请输出严格的 JSON 对象",
                raw=text,
            )
        problem = validate_tool_call(name, arguments, registry)
        if problem:
            return ParsedOutput(kind="invalid", thought=thought, reason=problem, raw=text)
        return ParsedOutput(
            kind="action", thought=thought, name=name, arguments=arguments, raw=text
        )
    return ParsedOutput(
        kind="invalid",
        thought=thought,
        reason="既没有 Action 也没有 Final Answer，请严格使用 ReAct 字段",
        raw=text,
    )


__all__ = [
    "ParsedOutput",
    "parse_react",
    "repair_json",
    "split_fields",
    "strip_markdown_fences",
    "validate_tool_call",
]

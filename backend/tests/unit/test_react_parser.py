from __future__ import annotations

from app.agent.react_parser import parse_react, repair_json, split_fields

REGISTRY = {
    "calculator": {
        "name": "calculator",
        "inputSchema": {
            "type": "object",
            "properties": {
                "expression": {"type": "string"},
                "precision": {"type": "integer", "minimum": 1},
            },
            "required": ["expression"],
            "additionalProperties": False,
        },
    },
    "weather": {
        "name": "weather",
        "inputSchema": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        },
    },
}


def test_final_answer_with_chinese_colon_and_noise() -> None:
    text = """
    好的，下面是结果。

    Thought：直接计算即可。

    Final Answer：结果是 0.3

    希望有帮助。
    """
    parsed = parse_react(text, REGISTRY)
    assert parsed.kind == "final"
    assert "0.3" in parsed.answer
    assert "希望" in parsed.answer


def test_action_with_markdown_fence_and_trailing_comma() -> None:
    text = """```text
Thought: 需要计算
Action: calculator
Action Input: ```json
{
  "expression": "0.1+0.2",
  "precision": 28,
}
```
```"""
    parsed = parse_react(text, REGISTRY)
    assert parsed.kind == "action"
    assert parsed.name == "calculator"
    assert parsed.arguments["expression"] == "0.1+0.2"


def test_action_input_with_prefix_text() -> None:
    text = (
        "前言说明。\nThought: 查天气\nAction: `weather`\n"
        'Action Input: 参数如下 {"location": "杭州"} 谢谢\n'
    )
    parsed = parse_react(text, REGISTRY)
    assert parsed.kind == "action"
    assert parsed.name == "weather"
    assert parsed.arguments == {"location": "杭州"}


def test_unknown_tool_is_invalid() -> None:
    parsed = parse_react("Thought: x\nAction: rm_rf\nAction Input: {}", REGISTRY)
    assert parsed.kind == "invalid"
    assert "未知工具" in parsed.reason


def test_schema_violation_is_invalid() -> None:
    parsed = parse_react(
        "Thought: x\nAction: calculator\nAction Input: {\"precision\": 5}", REGISTRY
    )
    assert parsed.kind == "invalid"
    assert "Schema" in parsed.reason


def test_missing_action_and_final_is_invalid() -> None:
    parsed = parse_react("Thought: 我先想想", REGISTRY)
    assert parsed.kind == "invalid"
    assert "既没有 Action 也没有 Final Answer" in parsed.reason


def test_json_repair_variants() -> None:
    assert repair_json('```json\n{"a": 1,}\n```') == {"a": 1}
    assert repair_json('说明 {"a": {"b": 2}} 结尾') == {"a": {"b": 2}}
    assert repair_json('{"a": 1') == {"a": 1}
    assert repair_json("not json") is None


def test_json_repair_keeps_braces_inside_strings() -> None:
    assert repair_json('{"query": "a } b"}') == {"query": "a } b"}


def test_split_fields_handles_observation_noise() -> None:
    fields = split_fields(
        "Thought: 计划\nAction: weather\nAction Input: {\"location\":\"杭州\"}\n"
        "Observation: 晴\nFinal Answer: 完成"
    )
    assert fields["thought"] == "计划"
    assert fields["action"] == "weather"
    assert "晴" in fields["observation"]
    assert fields["final answer"] == "完成"

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.context.state_card import compact_card_json
from app.models.db_models import StateCard

REACT_RULES = """你是极简可用的任务型 Agent。必须严格使用文本 ReAct 协议作答。

可用字段（每行一个字段，字段名固定）：
Thought: 你的推理与下一步计划
Action: 工具名（只能从工具清单里选）
Action Input: 一个合法 JSON 对象
（工具执行后系统会回填 Observation，你不必自己编造 Observation）
Final Answer: 面向用户的最终答案

规则：
1. 需要外部信息或计算时先输出 Action 与 Action Input，等待 Observation 后再继续。
2. 信息足够或任务完成时，输出 Final Answer，不要再输出 Action。
3. Action 必须是工具清单中的工具名；Action Input 必须符合该工具的 JSON Schema。
4. 不要输出原始思维链，只输出简洁的计划、决策与最终答案。
5. 工具失败时根据错误修正参数或改用其他工具，同一工具连续失败不要反复硬试。
6. 回复使用中文，除非用户指定其他语言。
"""

NATIVE_RULES = """你可以使用原生 tool_calls 调用工具，也可以直接给出最终答案。
仅在需要外部信息或计算时调用工具；调用失败时根据错误调整参数。回复使用中文。
"""


def build_system_prompt(registry: dict[str, dict[str, Any]], mode: str = "react") -> str:
    tool_lines = []
    for tool in registry.values():
        schema = json.dumps(tool.get("inputSchema", {}), ensure_ascii=False, sort_keys=True)
        tool_lines.append(
            f"- {tool['name']}: {tool.get('description', '')}\n  Schema: {schema}"
        )
    tools_text = "\n".join(tool_lines) or "- （当前没有可用工具）"
    rules = NATIVE_RULES if mode == "native" else REACT_RULES
    return f"{rules}\n\n工具清单：\n{tools_text}"


def read_agent_md(path: Path, limit: int = 20_000) -> str | None:
    if not path.is_file():
        return None
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    return content[:limit]


def agent_md_block(agent_md: str | None) -> dict[str, str] | None:
    if not agent_md:
        return None
    return {
        "role": "system",
        "content": "<agent_md>\n" + agent_md.strip() + "\n</agent_md>",
    }


def state_card_block(card: StateCard, slim: bool = False) -> dict[str, str]:
    return {
        "role": "system",
        "content": "状态卡（结构化上下文）：\n" + compact_card_json(card, slim=slim),
    }


def summary_block(summary: str) -> dict[str, str] | None:
    if not summary.strip():
        return None
    return {"role": "system", "content": "历史摘要：\n" + summary.strip()}


def native_tools_prompt(registry: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("inputSchema", {"type": "object", "properties": {}}),
            },
        }
        for tool in registry.values()
    ]


__all__ = [
    "agent_md_block",
    "build_system_prompt",
    "native_tools_prompt",
    "read_agent_md",
    "state_card_block",
    "summary_block",
]

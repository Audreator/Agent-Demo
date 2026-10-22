from __future__ import annotations

import json
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.db_models import StateCard


def card_to_dict(card: StateCard) -> dict[str, Any]:
    return {
        "goal": card.goal,
        "confirmed_facts": list(card.confirmed_facts or []),
        "decisions": list(card.decisions or []),
        "todos": list(card.todos or []),
        "artifacts": list(card.artifacts or []),
        "open_questions": list(card.open_questions or []),
        "confidence": card.confidence,
    }


def compact_card_json(card: StateCard, *, slim: bool = False) -> str:
    data = card_to_dict(card)
    if slim:
        data["confirmed_facts"] = data["confirmed_facts"][-6:]
        data["decisions"] = data["decisions"][-4:]
        data["todos"] = data["todos"][-6:]
        data["artifacts"] = data["artifacts"][-6:]
        data["open_questions"] = data["open_questions"][-3:]
        data["goal"] = data["goal"][:200]
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def _append_unique(items: list[Any], value: Any, limit: int) -> None:
    if not value:
        return
    if value in items:
        return
    items.append(value)
    if len(items) > limit:
        del items[: len(items) - limit]


class StateCardManager:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def get(self, session_id: UUID) -> StateCard:
        stmt = select(StateCard).where(StateCard.session_id == session_id)
        card = (await self.db.execute(stmt)).scalar_one_or_none()
        if card is None:
            card = StateCard(session_id=session_id)
            self.db.add(card)
            await self.db.flush()
        return card

    async def update_after_turn(
        self,
        session_id: UUID,
        *,
        turn_no: int,
        user_input: str,
        final_answer: str,
        tool_events: list[dict[str, Any]],
        errors: list[str],
    ) -> StateCard:
        card = await self.get(session_id)
        compact_input = _clip(user_input, 200)
        if not card.goal:
            card.goal = compact_input
        elif is_new_goal(user_input):
            card.goal = compact_input
        for event in tool_events:
            if event.get("status") == "ok":
                summary = _clip(str(event.get("summary") or ""), 160)
                if summary:
                    _append_unique(
                        card.confirmed_facts,
                        f"{event.get('name')}: {summary}",
                        12,
                    )
                ref = event.get("result_ref")
                if ref:
                    _append_unique(card.artifacts, ref, 12)
                if event.get("name") == "todo" and event.get("summary"):
                    _append_unique(card.todos, _clip(str(event["summary"]), 120), 12)
            else:
                _append_unique(
                    card.open_questions,
                    f"{event.get('name')} 执行失败: {_clip(str(event.get('error') or ''), 120)}",
                    8,
                )
        if final_answer:
            _append_unique(
                card.decisions,
                f"第{turn_no}轮结论: {_clip(final_answer, 160)}",
                10,
            )
        for error in errors:
            _append_unique(card.open_questions, _clip(f"待确认: {error}", 160), 8)
        if errors:
            card.confidence = min(card.confidence, 0.5)
        elif tool_events or final_answer:
            card.confidence = max(card.confidence, 0.75)
        card.version += 1
        await self.db.flush()
        return card


def _clip(text: str, limit: int) -> str:
    value = (text or "").strip().replace("\n", " ")
    return value[:limit] + ("…" if len(value) > limit else "")


_FOLLOW_UP_PREFIXES = (
    "继续",
    "请继续",
    "请再",
    "再",
    "然后",
    "刚才",
    "这个",
    "那个",
    "它",
    "上述",
    "接着",
)
_TASK_MARKERS = (
    "帮我",
    "请",
    "查询",
    "查一下",
    "搜索",
    "计算",
    "创建",
    "新建",
    "生成",
    "分析",
    "整理",
    "写",
    "做一个",
    "建一个",
)


def is_new_goal(user_input: str) -> bool:
    text = user_input.strip()
    if len(text) < 8 or text.startswith(_FOLLOW_UP_PREFIXES):
        return False
    if any(word in text[:8] for word in ("再", "继续", "然后", "刚才", "上面", "之前")):
        return False
    return any(marker in text for marker in _TASK_MARKERS)


__all__ = [
    "StateCardManager",
    "card_to_dict",
    "compact_card_json",
    "is_new_goal",
]

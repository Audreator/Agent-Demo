from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.prompt_builder import (
    agent_md_block,
    state_card_block,
    summary_block,
)
from app.config import Settings
from app.errors import BudgetExceededError
from app.models.db_models import Message, StateCard, Summary
from app.observability.tracing import Tracer


def estimate_tokens(text: str) -> int:
    return max(1, int(len(text) / 1.6))


def estimate_messages(messages: list[dict[str, str]]) -> int:
    return sum(estimate_tokens(str(message.get("content", ""))) for message in messages)


@dataclass
class ContextBundle:
    messages: list[dict[str, str]]
    token_est: int
    threshold: int
    compressed: bool = False
    tokens_before: int = 0
    tokens_after: int = 0
    summary_text: str = ""
    recent_turns: list[int] = field(default_factory=list)
    older_turns: list[int] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


class ContextManager:
    def __init__(
        self,
        db: AsyncSession,
        settings: Settings,
        *,
        user_id: str,
        tracer: Tracer | None = None,
    ) -> None:
        self.db = db
        self.settings = settings
        self.user_id = user_id
        self.tracer = tracer

    async def build(
        self,
        session_id: UUID,
        *,
        turn_no: int,
        user_input: str,
        base_system: str,
        agent_md: str | None,
    ) -> ContextBundle:
        messages = await self._load_messages(session_id, turn_no)
        card = await self._load_card(session_id)
        summaries = await self._load_summaries(session_id)
        grouped = _group_by_turn(messages)
        all_turns = sorted(grouped)
        recent_turns = all_turns[-self.settings.keep_recent_turns :]
        older_turns = [turn for turn in all_turns if turn not in recent_turns]

        covered_to = max((summary.covers_to_turn for summary in summaries), default=0)
        stored_summary = "\n".join(
            summary.content for summary in summaries if summary.content.strip()
        )
        new_summary_lines = [
            _summarize_turn(turn, grouped[turn])
            for turn in older_turns
            if turn > covered_to
        ]
        summary_text = "\n".join(
            part for part in [stored_summary, *new_summary_lines] if part.strip()
        )

        history = []
        for turn in recent_turns:
            history.extend(_render_turn(grouped[turn], self.settings.tool_result_context_limit))

        current_block = {"role": "user", "content": user_input}
        base_blocks = self._base_blocks(base_system, agent_md, card, summary_text, slim=False)
        messages_full = [*base_blocks, *history, current_block]
        tokens_before = estimate_messages(messages_full)
        threshold = self.settings.threshold

        compressed = False
        tokens_after = tokens_before
        if tokens_before > threshold:
            compressed = True
            trimmed_summary = _shrink_summary(summary_text)
            blocks = self._base_blocks(
                base_system, agent_md, card, trimmed_summary, slim=True
            )
            messages_full = [*blocks, *history, current_block]
            tokens_after = estimate_messages(messages_full)
            if tokens_after > threshold:
                blocks = self._base_blocks(
                    base_system,
                    agent_md,
                    card,
                    _shrink_summary(trimmed_summary, quarter=True),
                    slim=True,
                )
                messages_full = [*blocks, *history, current_block]
                tokens_after = estimate_messages(messages_full)
            if tokens_after > threshold:
                dropped_history = _render_recent_compactly(history)
                messages_full = [
                    *self._base_blocks(
                        base_system,
                        agent_md,
                        card,
                        _shrink_summary(summary_text, quarter=True),
                        slim=True,
                    ),
                    *dropped_history,
                    current_block,
                ]
                tokens_after = estimate_messages(messages_full)
            if tokens_after > threshold:
                raise BudgetExceededError(
                    f"上下文压缩后仍超预算：{tokens_after} > {threshold}"
                )
            await self._persist_incremental_summary(
                session_id, older_turns, covered_to, new_summary_lines, summary_text
            )

        if self.tracer is not None:
            await self.tracer.emit_event(
                "context_compress" if compressed else "context_assemble",
                {
                    "tokens_before": tokens_before,
                    "tokens_after": tokens_after,
                    "threshold": threshold,
                    "compressed": compressed,
                    "recent_turns": recent_turns,
                    "summarized_turns": older_turns,
                },
            )

        return ContextBundle(
            messages=messages_full,
            token_est=tokens_after,
            threshold=threshold,
            compressed=compressed,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            summary_text=summary_text,
            recent_turns=recent_turns,
            older_turns=older_turns,
            details={
                "message_count": len(messages_full),
                "kept_turns": recent_turns,
                "summarized_turns": older_turns,
            },
        )

    def _base_blocks(
        self,
        base_system: str,
        agent_md: str | None,
        card: StateCard,
        summary_text: str,
        *,
        slim: bool,
    ) -> list[dict[str, str]]:
        blocks: list[dict[str, str]] = [{"role": "system", "content": base_system}]
        agent_block = agent_md_block(agent_md)
        if agent_block:
            blocks.append(agent_block)
        blocks.append(state_card_block(card, slim=slim))
        summary = summary_block(summary_text)
        if summary:
            blocks.append(summary)
        return blocks

    async def _load_messages(self, session_id: UUID, turn_no: int) -> list[Message]:
        stmt = (
            select(Message)
            .where(Message.session_id == session_id, Message.turn_no < turn_no)
            .order_by(Message.turn_no.asc(), Message.created_at.asc())
        )
        return list((await self.db.execute(stmt)).scalars().all())

    async def _load_card(self, session_id: UUID) -> StateCard:
        stmt = select(StateCard).where(StateCard.session_id == session_id)
        card = (await self.db.execute(stmt)).scalar_one_or_none()
        if card is None:
            card = StateCard(session_id=session_id)
            self.db.add(card)
            await self.db.flush()
        return card

    async def _load_summaries(self, session_id: UUID) -> list[Summary]:
        stmt = (
            select(Summary)
            .where(Summary.session_id == session_id)
            .order_by(Summary.covers_from_turn.asc(), Summary.created_at.asc())
        )
        return list((await self.db.execute(stmt)).scalars().all())

    async def _persist_incremental_summary(
        self,
        session_id: UUID,
        older_turns: list[int],
        covered_to: int,
        new_summary_lines: list[str],
        summary_text: str,
    ) -> None:
        pending = [turn for turn in older_turns if turn > covered_to]
        if not pending or not new_summary_lines:
            return
        summary = Summary(
            session_id=session_id,
            covers_from_turn=min(pending),
            covers_to_turn=max(pending),
            content="\n".join(new_summary_lines),
            token_est=estimate_tokens(summary_text),
        )
        self.db.add(summary)
        await self.db.flush()


def _group_by_turn(messages: list[Message]) -> dict[int, list[Message]]:
    grouped: dict[int, list[Message]] = {}
    for message in messages:
        grouped.setdefault(message.turn_no, []).append(message)
    return grouped


def _render_turn(messages: list[Message], tool_limit: int) -> list[dict[str, str]]:
    rendered: list[dict[str, str]] = []
    for message in messages:
        if message.role == "user":
            rendered.append({"role": "user", "content": message.content})
        elif message.role == "assistant":
            rendered.append({"role": "assistant", "content": _render_assistant(message)})
        elif message.role == "tool":
            rendered.append({"role": "user", "content": _render_tool(message, tool_limit)})
        else:
            rendered.append({"role": "user", "content": message.content})
    return rendered


def _render_assistant(message: Message) -> str:
    meta = message.meta or {}
    if message.content.strip():
        return message.content
    plan = str(meta.get("plan") or "").strip()
    if plan:
        return "计划/假设：" + plan[:300]
    kind = str(meta.get("kind") or "")
    if kind in {"action", "parse_error", "wrap_up", "budget_exceeded"}:
        return ""
    thought = str(meta.get("thought") or "").strip()
    if thought:
        return "计划/假设：" + thought[:300]
    return ""


def _render_tool(message: Message, tool_limit: int) -> str:
    meta = message.meta or {}
    name = meta.get("tool_name", "tool")
    status = meta.get("status", "ok")
    result_ref = meta.get("result_ref")
    content = message.content or ""
    header = f"[工具结果 {name} status={status}]"
    if len(content) > tool_limit:
        summary = str(meta.get("summary") or content[:tool_limit])
        ref_line = f"\n完整结果见 {result_ref}" if result_ref else ""
        return f"{header}\n{summary}{ref_line}"
    if result_ref and status != "ok":
        return f"{header}\n{content}\n引用: {result_ref}"
    return f"{header}\n{content}"


def _render_recent_compactly(history: list[dict[str, str]]) -> list[dict[str, str]]:
    compacted = []
    for message in history:
        content = str(message.get("content", ""))
        refs = re.findall(r"artifact://[A-Za-z0-9_-]+", content)
        shortened = content[:400] + ("…" if len(content) > 400 else "")
        if refs and refs[0] not in shortened:
            shortened += f"\n完整结果见 {refs[0]}"
        compacted.append(
            {
                "role": message.get("role", "user"),
                "content": shortened,
            }
        )
    return compacted


def _summarize_turn(turn_no: int, messages: list[Message]) -> str:
    parts = [f"第{turn_no}轮"]
    for message in messages:
        if message.role == "user":
            parts.append(f"用户: {_clip(message.content, 200)}")
        elif message.role == "assistant":
            meta = message.meta or {}
            if message.content.strip():
                parts.append(f"助手: {_clip(message.content, 240)}")
            elif meta.get("plan"):
                parts.append(f"计划: {_clip(str(meta['plan']), 200)}")
        elif message.role == "tool":
            meta = message.meta or {}
            summary = meta.get("summary") or _clip(message.content, 140)
            ref = meta.get("result_ref")
            ref_part = f" 引用:{ref}" if ref else ""
            parts.append(
                f"工具 {meta.get('tool_name', 'tool')}({meta.get('status', 'ok')}): "
                f"{_clip(str(summary), 160)}{ref_part}"
            )
    return " | ".join(parts)


def _clip(text: str, limit: int) -> str:
    value = re.sub(r"\s+", " ", text or "").strip()
    return value[:limit] + ("…" if len(value) > limit else "")


def _shrink_summary(summary: str, *, quarter: bool = False) -> str:
    if not summary:
        return ""
    refs = list(dict.fromkeys(re.findall(r"artifact://[A-Za-z0-9_-]+", summary)))[:3]
    limit = 1200 if quarter else 4000
    if len(summary) <= limit:
        return summary
    lines = summary.splitlines()
    kept: list[str] = []
    size = 0
    for line in reversed(lines):
        if refs and any(ref in line for ref in refs):
            continue
        if size + len(line) > limit:
            break
        kept.append(line)
        size += len(line) + 1
    kept.reverse()
    if refs:
        kept.append("关键引用: " + " ".join(refs))
    return "\n".join(kept)


def tool_result_summary(result: Any, limit: int = 500) -> str:
    if isinstance(result, dict):
        for key in ("summary", "answer", "result", "content"):
            value = result.get(key)
            if isinstance(value, str) and value.strip():
                return value[:limit]
        return json.dumps(result, ensure_ascii=False)[:limit]
    return str(result)[:limit]


__all__ = [
    "ContextBundle",
    "ContextManager",
    "estimate_messages",
    "estimate_tokens",
    "tool_result_summary",
]

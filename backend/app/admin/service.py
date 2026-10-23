from __future__ import annotations

import math
from collections import Counter, defaultdict
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any
from uuid import UUID
from zoneinfo import ZoneInfo

from sqlalchemy import Float, Integer, String, Uuid, cast, exists, func, not_, select, tuple_
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.feedback import feedback_summaries
from app.config import Settings
from app.models.db_models import (
    Event,
    Message,
    MessageFeedback,
    Session,
    ToolCall,
)

TZ = ZoneInfo("Asia/Shanghai")
MAX_TEXT = 2000
MODEL_FAILURE_KINDS = (
    "model_transient_failure",
    "model_permanent_failure",
    "model_unknown_failure",
)


def parse_range(
    settings: Settings, start: str | None, end: str | None
) -> tuple[datetime, datetime]:
    now = datetime.now(UTC)
    end_dt = _parse_dt(end) if end else now
    start_dt = _parse_dt(start) if start else end_dt - timedelta(days=7)
    if end_dt <= start_dt:
        raise ValueError("start 必须早于 end")
    if end_dt - start_dt > timedelta(days=settings.admin_max_range_days):
        raise ValueError(f"查询范围不能超过 {settings.admin_max_range_days} 天")
    return start_dt, end_dt


def _parse_dt(value: str) -> datetime:
    text = str(value).strip().replace("Z", "+00:00")
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=TZ)
    return parsed.astimezone(UTC)


def percentile(values: list[int] | list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(p / 100 * len(ordered)) - 1)
    return float(ordered[min(index, len(ordered) - 1)])


def _tokens(meta: dict[str, Any] | None) -> int:
    payload = meta or {}
    usage = payload.get("usage")
    if isinstance(usage, dict):
        total = usage.get("total_tokens")
        if isinstance(total, (int, float)):
            return int(total)
        prompt = usage.get("prompt_tokens")
        completion = usage.get("completion_tokens")
        if isinstance(prompt, (int, float)) or isinstance(completion, (int, float)):
            return int(prompt or 0) + int(completion or 0)
    return 0


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(numerator / denominator, 4)


def _session_ids(user_id: str) -> Any:
    return select(cast(Session.id, String)).where(Session.user_id == user_id)


def _turn_conditions(
    settings: Settings,
    start: datetime,
    end: datetime,
    *,
    session_id: str | None = None,
    trace_id: str | None = None,
    tool_name: str | None = None,
    tool_status: str | None = None,
    rating: str | None = None,
    finish_reason: str | None = None,
) -> list[Any]:
    conditions: list[Any] = [
        Event.kind == "turn",
        Event.ts >= start,
        Event.ts < end,
        Event.session_id.is_not(None),
        Event.turn_no.is_not(None),
        Event.session_id.in_(_session_ids(settings.dev_user_id)),
    ]
    if session_id:
        conditions.append(Event.session_id == session_id)
    if trace_id:
        conditions.append(Event.trace_id == trace_id)
    if tool_name or tool_status:
        sub = select(ToolCall.id).where(
            ToolCall.session_id == cast(Event.session_id, Uuid),
            ToolCall.turn_no == Event.turn_no,
        )
        if tool_name:
            sub = sub.where(ToolCall.tool_name == tool_name)
        if tool_status:
            sub = sub.where(ToolCall.status == tool_status)
        conditions.append(exists(sub))
    if rating:
        sub = (
            select(MessageFeedback.id)
            .join(Message, Message.id == MessageFeedback.message_id)
            .where(
                Message.session_id == cast(Event.session_id, Uuid),
                Message.turn_no == Event.turn_no,
                MessageFeedback.user_id == settings.dev_user_id,
            )
        )
        if rating in {"like", "dislike"}:
            sub = sub.where(MessageFeedback.rating == rating)
            conditions.append(exists(sub))
        elif rating == "any":
            conditions.append(exists(sub))
        elif rating == "none":
            conditions.append(not_(exists(sub)))
    if finish_reason:
        kind, _, reason = finish_reason.partition(":")
        if kind in {
            "max_task_turns",
            "max_tool_iterations",
            "recoverable_limit",
            "max_model_calls",
        }:
            sub = select(Message.id).where(
                Message.session_id == cast(Event.session_id, Uuid),
                Message.turn_no == Event.turn_no,
                Message.role == "assistant",
                Message.meta["kind"].astext == "wrap_up",
                Message.meta["reason"].astext == kind,
            )
            conditions.append(exists(sub))
            return conditions
        if kind == "budget_exceeded":
            sub = select(Message.id).where(
                Message.session_id == cast(Event.session_id, Uuid),
                Message.turn_no == Event.turn_no,
                Message.role == "assistant",
                (
                    (Message.meta["kind"].astext == "budget_exceeded")
                    | (
                        (Message.meta["kind"].astext == "wrap_up")
                        & (Message.meta["reason"].astext == "budget_exceeded")
                    )
                ),
            )
            conditions.append(exists(sub))
            return conditions
        kind_map = {
            "final_answer": "final",
            "final": "final",
            "wrap_up": "wrap_up",
            "parse_error": "parse_error",
            "error": "error",
            "model_unavailable": "error",
        }
        mapped = kind_map.get(kind, kind)
        sub = select(Message.id).where(
            Message.session_id == cast(Event.session_id, Uuid),
            Message.turn_no == Event.turn_no,
            Message.role == "assistant",
            Message.meta["kind"].astext == mapped,
        )
        if reason:
            sub = sub.where(Message.meta["reason"].astext == reason)
        conditions.append(exists(sub))
    return conditions


async def _fetch_turn_events(
    db: AsyncSession,
    settings: Settings,
    start: datetime,
    end: datetime,
    *,
    session_id: str | None = None,
    trace_id: str | None = None,
    tool_name: str | None = None,
    tool_status: str | None = None,
    rating: str | None = None,
    finish_reason: str | None = None,
    limit: int | None = None,
    offset: int = 0,
) -> list[Any]:
    conditions = _turn_conditions(
        settings,
        start,
        end,
        session_id=session_id,
        trace_id=trace_id,
        tool_name=tool_name,
        tool_status=tool_status,
        rating=rating,
        finish_reason=finish_reason,
    )
    stmt = (
        select(Event.trace_id, Event.session_id, Event.turn_no, Event.ts, Event.duration_ms)
        .where(*conditions)
        .order_by(Event.ts.desc(), Event.id.desc())
    )
    if offset:
        stmt = stmt.offset(offset)
    if limit is not None:
        stmt = stmt.limit(limit)
    return list((await db.execute(stmt)).all())


async def _count_turns(
    db: AsyncSession, settings: Settings, start: datetime, end: datetime, **filters: Any
) -> int:
    conditions = _turn_conditions(settings, start, end, **filters)
    stmt = select(func.count()).select_from(Event).where(*conditions)
    return int((await db.execute(stmt)).scalar_one())


async def _turn_durations(
    db: AsyncSession, rows: list[Any]
) -> dict[str, int]:
    """按 trace 返回轮次耗时；缺失时用该 turn 首末消息时间差补齐。"""
    result: dict[str, int] = {}
    missing: list[tuple[UUID, int]] = []
    for row in rows:
        trace_id = str(row.trace_id)
        if row.duration_ms is not None and int(row.duration_ms) > 0:
            result[trace_id] = int(row.duration_ms)
            continue
        if not row.session_id or row.turn_no is None:
            continue
        try:
            missing.append((UUID(str(row.session_id)), int(row.turn_no)))
        except ValueError:
            continue
    if missing:
        message_rows = (
            await db.execute(
                select(Message.session_id, Message.turn_no, Message.created_at)
                .where(
                    tuple_(Message.session_id, Message.turn_no).in_(missing),
                )
                .order_by(Message.created_at.asc())
            )
        ).all()
        first_last: dict[tuple[str, int], list[datetime]] = defaultdict(list)
        for session_id, turn_no, created_at in message_rows:
            key = (str(session_id), int(turn_no))
            if len(first_last[key]) < 2:
                first_last[key].append(created_at)
            else:
                first_last[key][-1] = created_at
        duration_by_key = {
            key: max(0, int((items[-1] - items[0]).total_seconds() * 1000))
            for key, items in first_last.items()
            if items
        }
        for row in rows:
            trace_id = str(row.trace_id)
            if trace_id in result or not row.session_id or row.turn_no is None:
                continue
            result[trace_id] = duration_by_key.get(
                (str(row.session_id), int(row.turn_no)), 0
            )
    return result


async def _build_turn_rows(
    db: AsyncSession, settings: Settings, rows: list[Any]
) -> list[dict[str, Any]]:
    if not rows:
        return []
    keys: list[tuple[UUID, int]] = []
    session_ids: set[UUID] = set()
    for row in rows:
        if not row.session_id or row.turn_no is None:
            continue
        try:
            session_uuid = UUID(str(row.session_id))
        except ValueError:
            continue
        session_ids.add(session_uuid)
        keys.append((session_uuid, int(row.turn_no)))
    if not keys:
        return []
    session_rows = {
        session.id: session
        for session in (
            await db.execute(select(Session).where(Session.id.in_(session_ids)))
        )
        .scalars()
        .all()
    }
    messages = list(
        (
            await db.execute(
                select(Message).where(
                    Message.session_id.in_(session_ids),
                    tuple_(Message.session_id, Message.turn_no).in_(keys),
                )
            )
        )
        .scalars()
        .all()
    )
    tool_calls = list(
        (
            await db.execute(
                select(ToolCall).where(
                    ToolCall.session_id.in_(session_ids),
                    tuple_(ToolCall.session_id, ToolCall.turn_no).in_(keys),
                )
            )
        )
        .scalars()
        .all()
    )
    grouped_messages: dict[tuple[str, int], list[Message]] = defaultdict(list)
    for message in messages:
        grouped_messages[(str(message.session_id), message.turn_no)].append(message)
    for group in grouped_messages.values():
        group.sort(key=lambda item: item.created_at)
    grouped_tools: dict[tuple[str, int], list[ToolCall]] = defaultdict(list)
    for call in tool_calls:
        grouped_tools[(str(call.session_id), call.turn_no)].append(call)
    for tool_group in grouped_tools.values():
        tool_group.sort(key=lambda item: item.created_at)

    final_ids = []
    for group in grouped_messages.values():
        final = _final_message(group)
        if final is not None:
            final_ids.append(final.id)
    feedback_map = await feedback_summaries(db, final_ids, settings.dev_user_id)

    missing_layer_traces: set[str] = set()
    for row in rows:
        if not row.trace_id or row.turn_no is None:
            continue
        final = _final_message(
            grouped_messages.get((str(row.session_id), int(row.turn_no)), [])
        )
        if final is not None and not final.meta.get("layer"):
            missing_layer_traces.add(str(row.trace_id))
    layer_by_trace: dict[str, str] = {}
    if missing_layer_traces:
        layer_rows = (
            await db.execute(
                select(Event.trace_id, Event.payload, Event.ts)
                .where(
                    Event.trace_id.in_(missing_layer_traces),
                    Event.kind == "model_success",
                )
                .order_by(Event.ts.asc())
            )
        ).all()
        for trace_id, payload, _ts in layer_rows:
            layer = (payload or {}).get("layer")
            if layer:
                layer_by_trace[str(trace_id)] = str(layer)

    result: list[dict[str, Any]] = []
    for row in rows:
        if not row.session_id or row.turn_no is None:
            continue
        try:
            session_uuid = UUID(str(row.session_id))
        except ValueError:
            continue
        key = (str(session_uuid), int(row.turn_no))
        group = grouped_messages.get(key, [])
        calls = grouped_tools.get(key, [])
        user_message = next((item for item in group if item.role == "user"), None)
        final = _final_message(group)
        feedback = feedback_map.get(final.id) if final is not None else None
        started_at = user_message.created_at if user_message else row.ts
        finished_at = final.created_at if final is not None else row.ts
        duration = int(row.duration_ms or 0)
        if duration <= 0:
            duration = max(0, int((finished_at - started_at).total_seconds() * 1000))
        session = session_rows.get(session_uuid)
        layer = (final.meta.get("layer") if final is not None else None) or layer_by_trace.get(
            str(row.trace_id)
        )
        result.append(
            {
                "trace_id": str(row.trace_id),
                "session_id": str(session_uuid),
                "session_title": session.title if session else "",
                "turn_no": int(row.turn_no),
                "started_at": started_at.isoformat(),
                "user_input": (user_message.content if user_message else "")[:MAX_TEXT],
                "final_answer": (final.content if final is not None else "")[:MAX_TEXT],
                "model_layer": layer,
                "finish_reason": _finish_reason(final),
                "tokens": _tokens(final.meta if final is not None else None),
                "duration_ms": duration,
                "tools": [
                    {
                        "id": str(call.id),
                        "name": call.tool_name,
                        "status": call.status,
                        "latency_ms": int(call.latency_ms),
                        "retry_count": int(call.retry_count),
                        "error": call.error,
                        "summary": call.result_summary,
                        "result_ref": call.result_ref,
                        "arguments": call.arguments,
                        "result_meta": call.result_meta or {},
                    }
                    for call in calls
                ],
                "feedback": feedback.model_dump() if feedback is not None else None,
            }
        )
    return result


def _final_message(group: list[Message]) -> Message | None:
    for message in reversed(group):
        if message.role == "assistant" and message.meta.get("kind") in {"final", "wrap_up"}:
            return message
    for message in reversed(group):
        if message.role == "assistant":
            return message
    return None


def _finish_reason(message: Message | None) -> str:
    if message is None:
        return "unknown"
    kind = str(message.meta.get("kind") or "")
    if kind == "final":
        return "final_answer"
    if kind == "wrap_up":
        return f"wrap_up:{message.meta.get('reason') or ''}".rstrip(":")
    return kind or "unknown"


async def list_turns(
    db: AsyncSession,
    settings: Settings,
    start: datetime,
    end: datetime,
    *,
    page: int = 1,
    page_size: int = 50,
    **filters: Any,
) -> dict[str, Any]:
    total = await _count_turns(db, settings, start, end, **filters)
    offset = max(0, (page - 1) * page_size)
    page_rows = await _fetch_turn_events(
        db,
        settings,
        start,
        end,
        limit=page_size,
        offset=offset,
        **filters,
    )
    items = await _build_turn_rows(db, settings, page_rows)
    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "items": items,
    }


async def turn_detail(
    db: AsyncSession, settings: Settings, trace_id: str
) -> dict[str, Any] | None:
    events = list(
        (
            await db.execute(
                select(Event)
                .where(Event.trace_id == trace_id)
                .order_by(Event.ts.asc(), Event.id.asc())
            )
        )
        .scalars()
        .all()
    )
    if not events:
        return None
    located = next(
        (event for event in events if event.session_id and event.turn_no is not None), None
    )
    if located is None or not located.session_id:
        return None
    try:
        session_uuid = UUID(str(located.session_id))
    except ValueError:
        return None
    session = await db.get(Session, session_uuid)
    if session is None or session.user_id != settings.dev_user_id:
        return None
    row = SimpleNamespace(
        trace_id=trace_id,
        session_id=str(session_uuid),
        turn_no=int(located.turn_no or 0),
        ts=located.ts,
        duration_ms=next(
            (event.duration_ms for event in events if event.kind == "turn"), located.duration_ms
        )
        or 0,
    )
    built = await _build_turn_rows(db, settings, [row])
    summary = built[0] if built else {}
    return {
        "summary": summary,
        "messages": [
            {
                "id": str(message.id),
                "role": message.role,
                "turn_no": message.turn_no,
                "content": message.content[:10000],
                "meta": message.meta,
                "created_at": message.created_at.isoformat(),
            }
            for message in (
                await db.execute(
                    select(Message)
                    .where(
                        Message.session_id == session_uuid,
                        Message.turn_no == int(located.turn_no or 0),
                    )
                    .order_by(Message.created_at.asc())
                )
            )
            .scalars()
            .all()
        ],
        "tool_calls": summary.get("tools", []),
        "events": [
            {
                "id": event.id,
                "trace_id": event.trace_id,
                "span_id": event.span_id,
                "parent_span_id": event.parent_span_id,
                "kind": event.kind,
                "payload": event.payload,
                "duration_ms": event.duration_ms,
                "ts": event.ts.isoformat(),
            }
            for event in events
        ],
    }


async def _group_events(
    db: AsyncSession,
    settings: Settings,
    start: datetime,
    end: datetime,
    kind: str,
    field: str,
) -> dict[str, int]:
    expr = Event.payload[field].astext
    stmt = (
        select(expr, func.count())
        .where(
            Event.kind == kind,
            Event.ts >= start,
            Event.ts < end,
            Event.session_id.is_not(None),
            Event.session_id.in_(_session_ids(settings.dev_user_id)),
        )
        .group_by(expr)
    )
    return {
        str(key or "unknown"): int(count) for key, count in (await db.execute(stmt)).all()
    }


async def compute_alerts(
    db: AsyncSession, settings: Settings
) -> list[dict[str, Any]]:
    end = datetime.now(UTC)
    start = end - timedelta(hours=1)
    alerts: list[dict[str, Any]] = []

    tool_rows = (
        await db.execute(
            select(ToolCall.status)
            .join(Session, Session.id == ToolCall.session_id)
            .where(
                Session.user_id == settings.dev_user_id,
                ToolCall.created_at >= start,
                ToolCall.created_at < end,
            )
        )
    ).scalars().all()
    ok = sum(1 for status in tool_rows if status == "ok")
    errors = sum(1 for status in tool_rows if status == "error")
    tool_total = ok + errors
    failure_rate = errors / tool_total if tool_total else 0.0
    if (
        tool_total >= settings.admin_alert_min_tool_calls
        and failure_rate > settings.admin_alert_tool_failure_rate
    ):
        alerts.append(
            _alert(
                "tool_failure_rate",
                "工具失败率",
                round(failure_rate, 4),
                settings.admin_alert_tool_failure_rate,
                f"近 1 小时工具调用 {tool_total} 次，失败 {errors} 次",
                {"tool_status": "error", "range_hours": 1},
            )
        )

    ratings = (
        await db.execute(
            select(MessageFeedback.rating)
            .join(Session, Session.id == MessageFeedback.session_id)
            .where(
                Session.user_id == settings.dev_user_id,
                MessageFeedback.updated_at >= start,
                MessageFeedback.updated_at < end,
            )
        )
    ).scalars().all()
    likes = sum(1 for rating in ratings if rating == "like")
    dislikes = sum(1 for rating in ratings if rating == "dislike")
    feedback_total = likes + dislikes
    satisfaction = likes / feedback_total if feedback_total else 1.0
    if (
        feedback_total >= settings.admin_alert_min_feedbacks
        and satisfaction < settings.admin_alert_satisfaction_rate
    ):
        alerts.append(
            _alert(
                "satisfaction_rate",
                "满意率",
                round(satisfaction, 4),
                settings.admin_alert_satisfaction_rate,
                f"近 1 小时评价 {feedback_total} 条，赞 {likes} / 踩 {dislikes}",
                {"rating": "dislike", "range_hours": 1},
            )
        )

    duration_rows = (
        await db.execute(
            select(
                Event.trace_id,
                Event.session_id,
                Event.turn_no,
                Event.ts,
                Event.duration_ms,
            ).where(
                Event.kind == "turn",
                Event.ts >= start,
                Event.ts < end,
                Event.session_id.is_not(None),
                Event.session_id.in_(_session_ids(settings.dev_user_id)),
            )
        )
    ).all()
    duration_map = await _turn_durations(db, list(duration_rows))
    latency_values = [value for value in duration_map.values() if value > 0]
    p95 = percentile(latency_values, 95)
    if (
        len(latency_values) >= settings.admin_alert_min_turns
        and p95 is not None
        and p95 > settings.admin_alert_p95_ms
    ):
        alerts.append(
            _alert(
                "turn_p95",
                "轮次 P95 耗时",
                round(p95, 2),
                settings.admin_alert_p95_ms,
                f"近 1 小时 {len(latency_values)} 轮",
                {"range_hours": 1},
            )
        )

    breaker_count = (
        await db.execute(
            select(func.count())
            .select_from(Event)
            .where(
                Event.kind == "breaker_open",
                Event.ts >= start,
                Event.ts < end,
                Event.session_id.is_not(None),
                Event.session_id.in_(_session_ids(settings.dev_user_id)),
            )
        )
    ).scalar_one()
    if int(breaker_count) >= settings.admin_alert_breaker_count:
        alerts.append(
            _alert(
                "breaker_open",
                "熔断次数",
                int(breaker_count),
                settings.admin_alert_breaker_count,
                "近 1 小时发生模型熔断",
                {"range_hours": 1},
            )
        )

    wrap_count = (
        await db.execute(
            select(func.count())
            .select_from(Message)
            .join(Session, Session.id == Message.session_id)
            .where(
                Session.user_id == settings.dev_user_id,
                Message.created_at >= start,
                Message.created_at < end,
                (
                    (Message.meta["kind"].astext == "budget_exceeded")
                    | (
                        (Message.meta["kind"].astext == "wrap_up")
                        & Message.meta["reason"].astext.in_(
                            ["budget_exceeded", "max_task_turns"]
                        )
                    )
                ),
            )
        )
    ).scalar_one()
    if int(wrap_count) >= settings.admin_alert_wrap_count:
        alerts.append(
            _alert(
                "wrap_up",
                "预算/30 轮收束",
                int(wrap_count),
                settings.admin_alert_wrap_count,
                "近 1 小时预算收束或 30 轮触顶次数超限",
                {"finish_reason": "wrap_up", "range_hours": 1},
            )
        )
    return alerts


def _alert(
    rule: str,
    name: str,
    current: float | int,
    threshold: float | int,
    detail: str,
    filters: dict[str, Any],
) -> dict[str, Any]:
    return {
        "rule": rule,
        "name": name,
        "current": current,
        "threshold": threshold,
        "window": "最近 1 小时",
        "detail": detail,
        "filters": filters,
    }


def _bucket_key(value: datetime, hourly: bool) -> str:
    local = value.astimezone(TZ)
    if hourly:
        return local.replace(minute=0, second=0, microsecond=0).isoformat()
    return local.replace(hour=0, minute=0, second=0, microsecond=0).date().isoformat()


def _bucket_label(key: str, hourly: bool) -> str:
    if hourly:
        return datetime.fromisoformat(key).strftime("%m-%d %H:00")
    return datetime.fromisoformat(key).strftime("%m-%d")


def _buckets(start: datetime, end: datetime, hourly: bool) -> list[str]:
    local = start.astimezone(TZ)
    if hourly:
        cursor = local.replace(minute=0, second=0, microsecond=0)
        step = timedelta(hours=1)
    else:
        cursor = local.replace(hour=0, minute=0, second=0, microsecond=0)
        step = timedelta(days=1)
    keys: list[str] = []
    while cursor.astimezone(UTC) < end:
        keys.append(cursor.isoformat() if hourly else cursor.date().isoformat())
        cursor += step
    return keys


async def compute_metrics(
    db: AsyncSession, settings: Settings, start: datetime, end: datetime
) -> dict[str, Any]:
    turn_rows = await _fetch_turn_events(db, settings, start, end)
    session_ids = {
        UUID(str(row.session_id)) for row in turn_rows if row.session_id
    }
    session_rows = (
        list(
            (
                await db.execute(select(Session).where(Session.id.in_(session_ids)))
            )
            .scalars()
            .all()
        )
        if session_ids
        else []
    )
    tool_rows = list(
        (
            await db.execute(
                select(ToolCall)
                .join(Session, Session.id == ToolCall.session_id)
                .where(
                    Session.user_id == settings.dev_user_id,
                    ToolCall.created_at >= start,
                    ToolCall.created_at < end,
                )
            )
        )
        .scalars()
        .all()
    )
    feedback_rows = list(
        (
            await db.execute(
                select(MessageFeedback)
                .join(Session, Session.id == MessageFeedback.session_id)
                .where(
                    Session.user_id == settings.dev_user_id,
                    MessageFeedback.updated_at >= start,
                    MessageFeedback.updated_at < end,
                )
            )
        )
        .scalars()
        .all()
    )
    final_messages = list(
        (
            await db.execute(
                select(Message)
                .join(Session, Session.id == Message.session_id)
                .where(
                    Session.user_id == settings.dev_user_id,
                    Message.created_at >= start,
                    Message.created_at < end,
                    Message.role == "assistant",
                    Message.meta["kind"].astext.in_(["final", "wrap_up"]),
                )
            )
        )
        .scalars()
        .all()
    )

    turn_keys = {
        (str(row.session_id), int(row.turn_no))
        for row in turn_rows
        if row.session_id and row.turn_no is not None
    }
    tool_turn_keys = {(str(call.session_id), int(call.turn_no)) for call in tool_rows}
    task_keys = turn_keys & tool_turn_keys
    task_turns = len(task_keys)
    task_counts: Counter[str] = Counter(session_id for session_id, _ in task_keys)
    duration_map = await _turn_durations(db, turn_rows)
    durations = [value for value in duration_map.values() if value > 0]
    ok_calls = sum(1 for call in tool_rows if call.status == "ok")
    error_calls = sum(1 for call in tool_rows if call.status == "error")
    tool_rate = _rate(ok_calls, ok_calls + error_calls)
    likes = sum(1 for row in feedback_rows if row.rating == "like")
    dislikes = sum(1 for row in feedback_rows if row.rating == "dislike")
    satisfaction_rate = _rate(likes, likes + dislikes)
    total_tokens = sum(_tokens(message.meta) for message in final_messages)

    alerts = await compute_alerts(db, settings)
    overview = {
        "sessions": len(session_ids),
        "turns": len(turn_rows),
        "task_turns": task_turns,
        "tool_calls": len(tool_rows),
        "tool_success_rate": tool_rate,
        "avg_turn_latency_ms": round(sum(durations) / len(durations), 2) if durations else None,
        "p95_turn_latency_ms": percentile(durations, 95),
        "total_tokens": total_tokens,
        "likes": likes,
        "dislikes": dislikes,
        "satisfaction_rate": satisfaction_rate,
        "alerts": len(alerts),
    }

    hourly = (end - start) <= timedelta(days=2)
    bucket_keys = _buckets(start, end, hourly)
    turns_by_bucket: Counter[str] = Counter()
    for row in turn_rows:
        turns_by_bucket[_bucket_key(row.ts, hourly)] += 1
    tool_by_bucket: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for call in tool_rows:
        bucket = _bucket_key(call.created_at, hourly)
        if call.status == "ok":
            tool_by_bucket[bucket][0] += 1
        elif call.status == "error":
            tool_by_bucket[bucket][1] += 1
    feedback_by_bucket: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for row in feedback_rows:
        bucket = _bucket_key(row.updated_at, hourly)
        if row.rating == "like":
            feedback_by_bucket[bucket][0] += 1
        elif row.rating == "dislike":
            feedback_by_bucket[bucket][1] += 1
    latency_by_bucket: dict[str, list[int]] = defaultdict(list)
    for row in turn_rows:
        value = duration_map.get(str(row.trace_id), 0)
        if value > 0:
            latency_by_bucket[_bucket_key(row.ts, hourly)].append(value)
    trend = []
    for key in bucket_keys:
        tool_stats = tool_by_bucket.get(key, [0, 0])
        feedback_stats = feedback_by_bucket.get(key, [0, 0])
        trend.append(
            {
                "bucket": key,
                "label": _bucket_label(key, hourly),
                "turns": turns_by_bucket.get(key, 0),
                "tool_success_rate": _rate(tool_stats[0], tool_stats[0] + tool_stats[1]),
                "satisfaction_rate": _rate(
                    feedback_stats[0], feedback_stats[0] + feedback_stats[1]
                ),
                "p95_ms": percentile(latency_by_bucket.get(key, []), 95),
            }
        )

    tools = _tool_metrics(tool_rows)
    models = await _model_metrics(db, settings, start, end)
    context = await _context_metrics(db, settings, start, end)
    sessions = _session_metrics(session_rows, session_ids, task_counts)
    satisfaction = await _satisfaction_metrics(db, settings, feedback_rows, final_messages)

    return {
        "range": {
            "start": start.isoformat(),
            "end": end.isoformat(),
            "timezone": "Asia/Shanghai",
            "granularity": "hour" if hourly else "day",
        },
        "overview": overview,
        "trend": trend,
        "tools": tools,
        "models": models,
        "context": context,
        "sessions": sessions,
        "satisfaction": satisfaction,
        "alerts": alerts,
    }


def _tool_metrics(tool_rows: list[ToolCall]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for call in tool_rows:
        item = grouped.setdefault(
            call.tool_name,
            {
                "tool_name": call.tool_name,
                "calls": 0,
                "ok": 0,
                "error": 0,
                "blocked": 0,
                "latencies": [],
                "retries": 0,
                "errors": Counter(),
            },
        )
        item["calls"] += 1
        if call.status == "ok":
            item["ok"] += 1
        elif call.status == "error":
            item["error"] += 1
        elif call.status == "blocked":
            item["blocked"] += 1
        if call.status in {"ok", "error"}:
            item["latencies"].append(int(call.latency_ms))
        item["retries"] += int(call.retry_count)
        if call.status == "error" and call.error:
            item["errors"][call.error[:120]] += 1
    result: list[dict[str, Any]] = []
    for item in sorted(grouped.values(), key=lambda row: row["calls"], reverse=True):
        latencies = item["latencies"]
        result.append(
            {
                "tool_name": item["tool_name"],
                "calls": item["calls"],
                "ok": item["ok"],
                "error": item["error"],
                "blocked": item["blocked"],
                "success_rate": _rate(item["ok"], item["ok"] + item["error"]),
                "avg_ms": round(sum(latencies) / len(latencies), 2) if latencies else None,
                "p95_ms": percentile(latencies, 95),
                "retries": item["retries"],
                "top_errors": [
                    {"reason": reason, "count": count}
                    for reason, count in item["errors"].most_common(3)
                ],
            }
        )
    return result


async def _model_metrics(
    db: AsyncSession, settings: Settings, start: datetime, end: datetime
) -> list[dict[str, Any]]:
    calls = await _group_events(db, settings, start, end, "model_call", "layer")
    success = await _group_events(db, settings, start, end, "model_success", "layer")
    degrade = await _group_events(db, settings, start, end, "model_degrade", "from")
    breaker_open = await _group_events(db, settings, start, end, "breaker_open", "provider")
    half_open = await _group_events(db, settings, start, end, "breaker_half_open", "provider")
    probe_success = await _group_events(
        db, settings, start, end, "breaker_probe_success", "provider"
    )
    probe_failure = await _group_events(
        db, settings, start, end, "breaker_probe_failure", "provider"
    )
    failures: dict[str, int] = defaultdict(int)
    for kind in MODEL_FAILURE_KINDS:
        for layer, count in (
            await _group_events(db, settings, start, end, kind, "provider")
        ).items():
            failures[layer] += count

    layer_expr = Event.payload["layer"].astext
    avg_stmt = (
        select(
            layer_expr,
            func.avg(Event.duration_ms),
            func.count(),
            func.coalesce(
                func.sum(
                    cast(Event.payload["usage"]["total_tokens"].astext, Integer)
                ),
                0,
            ),
        )
        .where(
            Event.kind == "model_call",
            Event.ts >= start,
            Event.ts < end,
            Event.session_id.is_not(None),
            Event.session_id.in_(_session_ids(settings.dev_user_id)),
        )
        .group_by(layer_expr)
    )
    avg_map = {
        str(layer or "unknown"): (float(avg or 0), int(count or 0), int(tokens or 0))
        for layer, avg, count, tokens in (await db.execute(avg_stmt)).all()
    }
    success_layer_expr = Event.payload["layer"].astext
    success_tokens_stmt = (
        select(
            success_layer_expr,
            func.coalesce(
                func.sum(
                    cast(Event.payload["usage"]["total_tokens"].astext, Integer)
                ),
                0,
            ),
        )
        .where(
            Event.kind == "model_success",
            Event.ts >= start,
            Event.ts < end,
            Event.session_id.is_not(None),
            Event.session_id.in_(_session_ids(settings.dev_user_id)),
        )
        .group_by(success_layer_expr)
    )
    success_tokens = {
        str(layer or "unknown"): int(tokens or 0)
        for layer, tokens in (await db.execute(success_tokens_stmt)).all()
    }

    layers = sorted(
        set(calls) | set(success) | set(failures) | set(degrade) | set(breaker_open),
        key=lambda value: (
            ["deepseek", "qwen", "local"].index(value)
            if value in {"deepseek", "qwen", "local"}
            else 99
        ),
    )
    result: list[dict[str, Any]] = []
    for layer in layers:
        avg_ms, call_count, _ = avg_map.get(layer, (0.0, 0, 0))
        result.append(
            {
                "layer": layer,
                "calls": calls.get(layer, call_count),
                "ok": success.get(layer, 0),
                "error": failures.get(layer, 0),
                "degrade": degrade.get(layer, 0),
                "breaker_open": breaker_open.get(layer, 0),
                "half_open": half_open.get(layer, 0),
                "probe_success": probe_success.get(layer, 0),
                "probe_failure": probe_failure.get(layer, 0),
                "avg_ms": round(avg_ms, 2) if avg_ms else None,
                "tokens": success_tokens.get(layer, 0),
            }
        )
    return result


async def _context_metrics(
    db: AsyncSession, settings: Settings, start: datetime, end: datetime
) -> dict[str, Any]:
    compress = (
        await db.execute(
            select(
                func.count(),
                func.avg(cast(Event.payload["tokens_before"].astext, Float)),
                func.avg(cast(Event.payload["tokens_after"].astext, Float)),
                func.sum(cast(Event.payload["tokens_before"].astext, Float)),
                func.sum(cast(Event.payload["tokens_after"].astext, Float)),
            ).where(
                Event.kind == "context_compress",
                Event.ts >= start,
                Event.ts < end,
                Event.session_id.is_not(None),
                Event.session_id.in_(_session_ids(settings.dev_user_id)),
            )
        )
    ).one()
    compress_count, before_avg, after_avg, before_sum, after_sum = compress

    async def _message_count(*conditions: Any) -> int:
        stmt = (
            select(func.count())
            .select_from(Message)
            .join(Session, Session.id == Message.session_id)
            .where(
                Session.user_id == settings.dev_user_id,
                Message.created_at >= start,
                Message.created_at < end,
                *conditions,
            )
        )
        return int((await db.execute(stmt)).scalar_one())

    budget_count = await _message_count(Message.meta["kind"].astext == "budget_exceeded")
    wrap_budget = await _message_count(
        Message.meta["kind"].astext == "wrap_up",
        Message.meta["reason"].astext == "budget_exceeded",
    )
    max_turn_wraps = await _message_count(
        Message.meta["kind"].astext == "wrap_up",
        Message.meta["reason"].astext == "max_task_turns",
    )
    wrap_message_sessions = set(
        (
            await db.execute(
                select(Message.session_id)
                .join(Session, Session.id == Message.session_id)
                .where(
                    Session.user_id == settings.dev_user_id,
                    Message.created_at >= start,
                    Message.created_at < end,
                    Message.meta["kind"].astext == "wrap_up",
                    Message.meta["reason"].astext == "max_task_turns",
                )
            )
        )
        .scalars()
        .all()
    )
    wrapped_status_sessions = set(
        (
            await db.execute(
                select(Session.id).where(
                    Session.user_id == settings.dev_user_id,
                    Session.status == "wrapped",
                    Session.updated_at >= start,
                    Session.updated_at < end,
                )
            )
        )
        .scalars()
        .all()
    )
    max_turns = max_turn_wraps + len(wrapped_status_sessions - wrap_message_sessions)
    mode_switch = await _group_events(db, settings, start, end, "mode_switch", "to")
    return {
        "compress_count": int(compress_count or 0),
        "tokens_before_avg": round(float(before_avg), 2) if before_avg is not None else None,
        "tokens_after_avg": round(float(after_avg), 2) if after_avg is not None else None,
        "tokens_before_total": int(before_sum or 0),
        "tokens_after_total": int(after_sum or 0),
        "budget_wrap_count": budget_count + wrap_budget,
        "max_turn_wrap_count": max_turns,
        "mode_switch_count": sum(mode_switch.values()),
    }


def _session_metrics(
    sessions: list[Session],
    session_ids: set[UUID],
    task_counts: Counter[str],
) -> dict[str, Any]:
    counts = [task_counts.get(str(session.id), 0) for session in sessions]
    active_counts = [count for count in counts if count > 0]
    distribution = {"1-5": 0, "6-10": 0, "11-20": 0, "21-30": 0, "30+": 0}
    for count in active_counts:
        if count <= 5:
            distribution["1-5"] += 1
        elif count <= 10:
            distribution["6-10"] += 1
        elif count <= 20:
            distribution["11-20"] += 1
        elif count <= 30:
            distribution["21-30"] += 1
        else:
            distribution["30+"] += 1
    return {
        "count": len(session_ids),
        "avg_task_turns": round(sum(active_counts) / len(active_counts), 2)
        if active_counts
        else 0,
        "distribution": distribution,
        "wrapped": sum(1 for session in sessions if session.status == "wrapped"),
        "native": sum(1 for session in sessions if session.mode == "native"),
    }


async def _satisfaction_metrics(
    db: AsyncSession,
    settings: Settings,
    feedback_rows: list[MessageFeedback],
    final_messages: list[Message],
) -> dict[str, Any]:
    likes = sum(1 for row in feedback_rows if row.rating == "like")
    dislikes = sum(1 for row in feedback_rows if row.rating == "dislike")
    reasons: Counter[str] = Counter(
        row.reason_code for row in feedback_rows if row.reason_code
    )
    message_ids = [row.message_id for row in feedback_rows]
    message_map: dict[UUID, Message] = {}
    if message_ids:
        for message in (
            await db.execute(select(Message).where(Message.id.in_(message_ids)))
        ).scalars().all():
            message_map[message.id] = message
    turns = {
        (str(message.session_id), int(message.turn_no)): message
        for message in message_map.values()
    }
    tool_turns: dict[tuple[str, int], set[str]] = defaultdict(set)
    if turns:
        call_rows = (
            await db.execute(
                select(ToolCall.session_id, ToolCall.turn_no, ToolCall.tool_name).where(
                    tuple_(ToolCall.session_id, ToolCall.turn_no).in_(
                        [(message.session_id, message.turn_no) for message in message_map.values()]
                    )
                )
            )
        ).all()
        for session_id, turn_no, tool_name in call_rows:
            tool_turns[(str(session_id), int(turn_no))].add(tool_name)

    by_tool: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for row in feedback_rows:
        current_message = message_map.get(row.message_id)
        if current_message is None:
            continue
        key = (str(current_message.session_id), int(current_message.turn_no))
        for tool_name in tool_turns.get(key, set()):
            if row.rating == "like":
                by_tool[tool_name][0] += 1
            elif row.rating == "dislike":
                by_tool[tool_name][1] += 1

    traces = {row.trace_id for row in feedback_rows if row.trace_id}
    layer_by_trace: dict[str, str] = {}
    if traces:
        layer_rows = (
            await db.execute(
                select(Event.trace_id, Event.payload, Event.ts)
                .where(Event.trace_id.in_(traces), Event.kind == "model_success")
                .order_by(Event.ts.asc())
            )
        ).all()
        for trace_id, payload, _ts in layer_rows:
            layer = (payload or {}).get("layer")
            if layer:
                layer_by_trace[str(trace_id)] = str(layer)
    by_model: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for row in feedback_rows:
        layer = None
        current_message = message_map.get(row.message_id)
        if current_message is not None:
            layer = current_message.meta.get("layer")
        if not layer and row.trace_id:
            layer = layer_by_trace.get(row.trace_id)
        layer = str(layer or "unknown")
        if row.rating == "like":
            by_model[layer][0] += 1
        elif row.rating == "dislike":
            by_model[layer][1] += 1

    return {
        "total": likes + dislikes,
        "likes": likes,
        "dislikes": dislikes,
        "rate": _rate(likes, likes + dislikes),
        "reasons": [
            {"code": code, "count": count} for code, count in reasons.most_common()
        ],
        "by_tool": [
            {
                "tool_name": name,
                "likes": stats[0],
                "dislikes": stats[1],
                "rate": _rate(stats[0], stats[0] + stats[1]),
            }
            for name, stats in sorted(
                by_tool.items(), key=lambda item: item[1][0] + item[1][1], reverse=True
            )
        ],
        "by_model": [
            {
                "layer": layer,
                "likes": stats[0],
                "dislikes": stats[1],
                "rate": _rate(stats[0], stats[0] + stats[1]),
            }
            for layer, stats in sorted(by_model.items())
        ],
    }


__all__ = [
    "compute_alerts",
    "compute_metrics",
    "list_turns",
    "parse_range",
    "percentile",
    "turn_detail",
]

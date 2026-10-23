from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.db import get_db
from app.models.db_models import Message, MessageFeedback, Session, utcnow
from app.sessions.schemas import FeedbackIn, FeedbackOut, FeedbackSummaryOut

router = APIRouter()
DbSession = Annotated[AsyncSession, Depends(get_db)]

FEEDBACK_REASONS = (
    "irrelevant",
    "tool_error",
    "bad_link",
    "too_slow",
    "fact_error",
    "incomplete",
    "other",
)
RATINGS = ("like", "dislike")


def _settings(request: Request) -> Settings:
    return request.app.state.app_state.settings


async def feedback_summaries(
    db: AsyncSession, message_ids: list[UUID], user_id: str
) -> dict[UUID, FeedbackSummaryOut]:
    if not message_ids:
        return {}
    counts_stmt = (
        select(MessageFeedback.message_id, MessageFeedback.rating, func.count())
        .where(MessageFeedback.message_id.in_(message_ids))
        .group_by(MessageFeedback.message_id, MessageFeedback.rating)
    )
    summary: dict[UUID, FeedbackSummaryOut] = {}
    for message_id, rating, count in (await db.execute(counts_stmt)).all():
        item = summary.setdefault(message_id, FeedbackSummaryOut())
        if rating == "like":
            item.likes = int(count)
        elif rating == "dislike":
            item.dislikes = int(count)
    current_stmt = select(MessageFeedback).where(
        MessageFeedback.message_id.in_(message_ids),
        MessageFeedback.user_id == user_id,
    )
    for row in (await db.execute(current_stmt)).scalars().all():
        item = summary.setdefault(row.message_id, FeedbackSummaryOut())
        item.rating = row.rating
        item.reason_code = row.reason_code
        item.reason_text = row.reason_text
        item.updated_at = row.updated_at
    return summary


async def like_dislike_counts(db: AsyncSession, message_id: UUID) -> tuple[int, int]:
    stmt = (
        select(MessageFeedback.rating, func.count())
        .where(MessageFeedback.message_id == message_id)
        .group_by(MessageFeedback.rating)
    )
    likes = dislikes = 0
    for rating, count in (await db.execute(stmt)).all():
        if rating == "like":
            likes = int(count)
        elif rating == "dislike":
            dislikes = int(count)
    return likes, dislikes


async def _load_message(
    db: AsyncSession, session_id: UUID, message_id: UUID, user_id: str
) -> tuple[Session, Message]:
    session = await db.get(Session, session_id)
    if session is None or session.user_id != user_id:
        raise HTTPException(status_code=404, detail="会话不存在")
    message = await db.get(Message, message_id)
    if message is None or message.session_id != session_id:
        raise HTTPException(status_code=404, detail="消息不存在")
    return session, message


@router.post(
    "/sessions/{session_id}/messages/{message_id}/feedback",
    response_model=FeedbackOut,
)
async def submit_feedback(
    session_id: UUID,
    message_id: UUID,
    body: FeedbackIn,
    db: DbSession,
    request: Request,
) -> FeedbackOut:
    settings = _settings(request)
    user_id = settings.dev_user_id
    session, message = await _load_message(db, session_id, message_id, user_id)
    if message.role != "assistant" or message.meta.get("kind") not in {"final", "wrap_up"}:
        raise HTTPException(status_code=400, detail="仅最终回答支持评价")

    rating = body.rating.lower()
    if rating not in RATINGS:
        raise HTTPException(status_code=422, detail="rating 必须是 like 或 dislike")

    reason_code: str | None = None
    reason_text: str | None = None
    if rating == "dislike":
        reason_code = (body.reason_code or "").strip()
        if reason_code not in FEEDBACK_REASONS:
            raise HTTPException(status_code=422, detail="原因编码非法")
        if reason_code == "other":
            reason_text = (body.reason_text or "").strip()
            max_len = min(settings.feedback_reason_max, 200)
            if not 1 <= len(reason_text) <= max_len:
                raise HTTPException(
                    status_code=422, detail=f"选择其他时必须填写 1-{max_len} 字说明"
                )
        else:
            reason_text = None
    elif body.reason_code or body.reason_text:
        raise HTTPException(status_code=422, detail="点赞时不应携带原因")

    trace_id = message.meta.get("trace_id") or session.last_trace_id
    now = utcnow()
    stmt = (
        pg_insert(MessageFeedback)
        .values(
            message_id=message_id,
            session_id=session_id,
            user_id=user_id,
            rating=rating,
            reason_code=reason_code,
            reason_text=reason_text,
            trace_id=trace_id,
            created_at=now,
            updated_at=now,
        )
        .on_conflict_do_update(
            constraint="uq_feedback_message_user",
            set_={
                "rating": rating,
                "reason_code": reason_code,
                "reason_text": reason_text,
                "trace_id": trace_id,
                "updated_at": now,
            },
        )
        .returning(MessageFeedback.updated_at)
    )
    try:
        result = await db.execute(stmt)
        updated_at = result.scalar_one()
        await db.commit()
    except IntegrityError:
        await db.rollback()
        row = (
            await db.execute(
                select(MessageFeedback).where(
                    MessageFeedback.message_id == message_id,
                    MessageFeedback.user_id == user_id,
                )
            )
        ).scalar_one()
        rating = row.rating
        reason_code = row.reason_code
        reason_text = row.reason_text
        updated_at = row.updated_at
    likes, dislikes = await like_dislike_counts(db, message_id)
    return FeedbackOut(
        message_id=message_id,
        rating=rating,
        reason_code=reason_code,
        reason_text=reason_text,
        likes=likes,
        dislikes=dislikes,
        updated_at=updated_at,
    )


@router.delete(
    "/sessions/{session_id}/messages/{message_id}/feedback",
    status_code=204,
)
async def cancel_feedback(
    session_id: UUID,
    message_id: UUID,
    db: DbSession,
    request: Request,
) -> Response:
    settings = _settings(request)
    user_id = settings.dev_user_id
    await _load_message(db, session_id, message_id, user_id)
    await db.execute(
        delete(MessageFeedback).where(
            MessageFeedback.message_id == message_id,
            MessageFeedback.user_id == user_id,
        )
    )
    await db.commit()
    return Response(status_code=204)


__all__ = [
    "FEEDBACK_REASONS",
    "feedback_summaries",
    "like_dislike_counts",
    "router",
]

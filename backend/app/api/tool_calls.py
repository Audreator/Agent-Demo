from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.db import get_db
from app.models.db_models import Artifact, Message, Session, ToolCall
from app.sessions.schemas import ToolCallOut, ToolResultOut

router = APIRouter()
DbSession = Annotated[AsyncSession, Depends(get_db)]


def _settings(request: Request) -> Settings:
    return request.app.state.app_state.settings


async def _require_session(
    db: AsyncSession, session_id: UUID, settings: Settings
) -> Session:
    session = await db.get(Session, session_id)
    if session is None or session.user_id != settings.dev_user_id:
        raise HTTPException(status_code=404, detail="会话不存在")
    return session


@router.get("/sessions/{session_id}/tool-calls", response_model=list[ToolCallOut])
async def list_tool_calls(
    session_id: UUID,
    db: DbSession,
    request: Request,
    turn_no: int | None = None,
) -> list[ToolCall]:
    settings = _settings(request)
    await _require_session(db, session_id, settings)
    stmt = select(ToolCall).where(ToolCall.session_id == session_id)
    if turn_no is not None:
        stmt = stmt.where(ToolCall.turn_no == turn_no)
    stmt = stmt.order_by(ToolCall.turn_no.asc(), ToolCall.created_at.asc(), ToolCall.id.asc())
    return list((await db.execute(stmt)).scalars().all())


@router.get(
    "/sessions/{session_id}/tool-calls/{tool_call_id}/result",
    response_model=ToolResultOut,
)
async def get_tool_call_result(
    session_id: UUID,
    tool_call_id: UUID,
    db: DbSession,
    request: Request,
) -> ToolResultOut:
    settings = _settings(request)
    await _require_session(db, session_id, settings)
    call = await db.get(ToolCall, tool_call_id)
    if call is None or call.session_id != session_id:
        raise HTTPException(status_code=404, detail="工具调用不存在")

    artifact_id = None
    if call.result_ref and call.result_ref.startswith("artifact://"):
        artifact_id = call.result_ref.removeprefix("artifact://")
    if artifact_id:
        artifact = await db.get(Artifact, artifact_id)
        if artifact is not None and artifact.session_id == session_id:
            return ToolResultOut(
                tool_call_id=tool_call_id,
                content=artifact.content,
                source="artifact",
                result_ref=call.result_ref,
            )

    stmt = (
        select(Message)
        .where(
            Message.session_id == session_id,
            Message.meta["tool_call_id"].astext.in_([call.id.hex, str(call.id)]),
        )
        .order_by(Message.created_at.desc())
        .limit(1)
    )
    message = (await db.execute(stmt)).scalars().first()
    if message is None:
        candidates_stmt = (
            select(Message)
            .where(
                Message.session_id == session_id,
                Message.turn_no == call.turn_no,
                Message.role == "tool",
                Message.meta["tool_name"].astext == call.tool_name,
            )
            .order_by(Message.created_at.asc())
        )
        candidates = list((await db.execute(candidates_stmt)).scalars().all())
        if candidates:
            message = min(
                candidates,
                key=lambda item: abs((item.created_at - call.created_at).total_seconds()),
            )
    if message is not None:
        return ToolResultOut(
            tool_call_id=tool_call_id,
            content=message.content,
            source="message",
            result_ref=call.result_ref,
        )
    raise HTTPException(status_code=404, detail="完整结果不存在")


__all__ = ["router"]

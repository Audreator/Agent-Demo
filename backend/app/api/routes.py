from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.sse import SSEEmitter, encode_sse, replay_events, stream_buffer
from app.db import get_db
from app.models.db_models import Event, Message, Session
from app.sessions.repository import SessionRepository
from app.sessions.schemas import (
    MessageOut,
    SendMessageRequest,
    SessionCreate,
    SessionOut,
    ToolInfo,
    TraceEvent,
)

logger = logging.getLogger("agent.api")
router = APIRouter()
DbSession = Annotated[AsyncSession, Depends(get_db)]


def _state(request: Request) -> Any:
    return request.app.state.app_state


def _repo(db: AsyncSession, request: Request) -> SessionRepository:
    state = _state(request)
    return SessionRepository(
        db, state.settings.dev_user_id, artifacts_dir=state.settings.artifacts_dir
    )


@router.get("/health")
async def health(request: Request) -> dict[str, Any]:
    state = _state(request)
    mcp_started = state.mcp.started
    model_ready = state.gateway.has_available_model()
    return {
        "status": "ok" if mcp_started and model_ready else "degraded",
        "models": state.gateway.status(),
        "mcp": {
            "started": mcp_started,
            "pid": state.mcp.process_id,
            "tools": [tool.get("name") for tool in state.tools],
        },
    }


@router.get("/sessions", response_model=list[SessionOut])
async def list_sessions(db: DbSession, request: Request) -> list[Session]:
    repo = _repo(db, request)
    return await repo.list_sessions()


@router.post("/sessions", response_model=SessionOut, status_code=201)
async def create_session(
    db: DbSession, request: Request, body: SessionCreate | None = None
) -> Session:
    repo = _repo(db, request)
    title = (body.title if body else None) or "新会话"
    session = await repo.create(title=title)
    await db.commit()
    return session


@router.get("/sessions/{session_id}/messages", response_model=list[MessageOut])
async def list_messages(
    session_id: UUID, db: DbSession, request: Request
) -> list[Message]:
    repo = _repo(db, request)
    session = await repo.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    return await repo.list_messages(session_id)


@router.delete("/sessions/{session_id}", status_code=204)
async def delete_session(session_id: UUID, db: DbSession, request: Request) -> Response:
    state = _state(request)
    active = state.streams.get(session_id)
    if active is not None and not active.completed:
        raise HTTPException(status_code=409, detail="该会话有正在执行的轮次，暂不能删除")
    repo = _repo(db, request)
    deleted = await repo.delete(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="会话不存在")
    state.streams.remove(session_id)
    state.session_locks.pop(str(session_id), None)
    await db.commit()
    return Response(status_code=204)


@router.get("/tools", response_model=list[ToolInfo])
async def list_tools(request: Request) -> list[dict[str, Any]]:
    state = _state(request)
    if not state.tools:
        try:
            state.tools = await state.mcp.list_tools()
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"工具服务不可用: {exc}") from exc
    return [
        {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "inputSchema": tool.get("inputSchema", {}),
        }
        for tool in state.tools
    ]


@router.get("/traces/{trace_id}", response_model=list[TraceEvent])
async def get_trace(trace_id: str, db: DbSession) -> list[Event]:
    stmt = (
        select(Event)
        .where(Event.trace_id == trace_id)
        .order_by(Event.ts.asc(), Event.id.asc())
    )
    return list((await db.execute(stmt)).scalars().all())


@router.post("/sessions/{session_id}/messages")
async def send_message(
    session_id: UUID,
    body: SendMessageRequest,
    db: DbSession,
    request: Request,
) -> StreamingResponse:
    state = _state(request)
    repo = _repo(db, request)
    session = await repo.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    last_event_id = body.last_event_id
    if last_event_id is None:
        raw_last_event_id = request.headers.get("Last-Event-ID") or request.query_params.get(
            "last_event_id"
        )
        if raw_last_event_id:
            try:
                last_event_id = int(raw_last_event_id)
            except ValueError:
                last_event_id = None
    if last_event_id is not None and last_event_id > 0:
        buffer = state.streams.get(session_id)
        if (
            buffer is not None
            and body.client_message_id
            and buffer.client_message_id
            and body.client_message_id != buffer.client_message_id
        ):
            buffer = None
        if buffer is not None:
            events = replay_events(buffer, last_event_id)
            await db.close()
            if buffer.completed and not events:
                return StreamingResponse(
                    _replay_generator(
                        [
                            (
                                "done",
                                {
                                    "seq": last_event_id + 1,
                                    "finish_reason": "already_completed",
                                },
                            )
                        ]
                    ),
                    media_type="text/event-stream",
                    headers=_sse_headers(),
                )
            return StreamingResponse(
                stream_buffer(buffer, start_after_seq=last_event_id),
                media_type="text/event-stream",
                headers=_sse_headers(),
            )
        await db.close()
        return StreamingResponse(
            _replay_generator(
                [
                    (
                        "error",
                        {
                            "seq": last_event_id + 1,
                            "code": "RESUME_UNAVAILABLE",
                            "message": "无法续传：服务端已丢失该流的缓冲，请重新发送消息。",
                            "class": "state",
                            "retryable": False,
                        },
                    ),
                    (
                        "done",
                        {
                            "seq": last_event_id + 2,
                            "finish_reason": "resume_unavailable",
                        },
                    ),
                ]
            ),
            media_type="text/event-stream",
            headers=_sse_headers(),
        )

    active = state.streams.get(session_id)
    if active is not None:
        same_request = (
            body.client_message_id is not None
            and active.client_message_id == body.client_message_id
        )
        same_text = (
            body.client_message_id is None
            and not active.completed
            and active.request_text == body.content
        )
        if same_request or same_text:
            await db.close()
            return StreamingResponse(
                stream_buffer(active), media_type="text/event-stream", headers=_sse_headers()
            )
        if not active.completed:
            await db.close()
            raise HTTPException(status_code=409, detail="该会话已有正在执行的轮次，请等待完成")
    if body.client_message_id:
        duplicate_stmt = (
            select(Message.id)
            .where(
                Message.session_id == session_id,
                Message.meta["client_message_id"].astext == body.client_message_id,
            )
            .limit(1)
        )
        if (await db.execute(duplicate_stmt)).first() is not None:
            await db.close()
            return StreamingResponse(
                _replay_generator(
                    [
                        (
                            "error",
                            {
                                "seq": 1,
                                "code": "DUPLICATE_REQUEST",
                                "message": "该消息已处理过，请刷新会话查看结果。",
                                "class": "state",
                                "retryable": False,
                            },
                        ),
                        (
                            "done",
                            {"seq": 2, "finish_reason": "duplicate_request"},
                        ),
                    ]
                ),
                media_type="text/event-stream",
                headers=_sse_headers(),
            )
    await db.close()
    buffer = state.streams.create(
        session_id, body.content, body.client_message_id
    )
    emitter = SSEEmitter(buffer)
    lock = state.session_locks.setdefault(str(session_id), asyncio.Lock())
    trace_id = state.loop.tracer.new_trace()

    async def runner() -> None:
        async with lock:
            try:
                await state.loop.run(
                    session_id,
                    body.content,
                    emitter.emit,
                    trace_id=trace_id,
                    client_message_id=body.client_message_id,
                )
            except Exception as exc:  # defensive: loop already handles AgentError
                logger.exception("agent loop crashed")
                await emitter.emit(
                    "error",
                    {
                        "code": "INTERNAL_ERROR",
                        "message": f"服务内部错误: {exc}",
                        "class": "state",
                        "retryable": False,
                    },
                )
                await emitter.emit("done", {"finish_reason": "error"})
            finally:
                buffer.finish()

    buffer.task = asyncio.create_task(runner())
    return StreamingResponse(
        stream_buffer(buffer), media_type="text/event-stream", headers=_sse_headers()
    )


def _sse_headers() -> dict[str, str]:
    return {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }


async def _replay_generator(events: list[tuple[str, dict[str, Any]]]) -> AsyncIterator[str]:
    for event, payload in events:
        yield encode_sse(event, payload)


__all__ = ["router"]

from __future__ import annotations

from pathlib import Path
from uuid import UUID

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.db_models import (
    Artifact,
    Message,
    Session,
    Snapshot,
    StateCard,
    Summary,
    Todo,
    ToolCall,
    User,
)


class SessionRepository:
    def __init__(
        self, db: AsyncSession, user_id: str, *, artifacts_dir: Path | None = None
    ) -> None:
        self.db = db
        self.user_id = user_id
        self.artifacts_dir = artifacts_dir

    async def ensure_user(self) -> None:
        existing = await self.db.get(User, self.user_id)
        if existing is None:
            self.db.add(User(id=self.user_id, name=self.user_id))
            await self.db.flush()

    async def create(self, title: str = "新会话") -> Session:
        await self.ensure_user()
        session = Session(user_id=self.user_id, title=title)
        self.db.add(session)
        await self.db.flush()
        self.db.add(StateCard(session_id=session.id))
        await self.db.flush()
        return session

    async def list_sessions(self) -> list[Session]:
        stmt = (
            select(Session)
            .where(Session.user_id == self.user_id)
            .order_by(Session.updated_at.desc())
        )
        return list((await self.db.execute(stmt)).scalars().all())

    async def get(self, session_id: UUID) -> Session | None:
        stmt = select(Session).where(
            Session.id == session_id, Session.user_id == self.user_id
        )
        return (await self.db.execute(stmt)).scalar_one_or_none()

    async def get_state_card(self, session_id: UUID) -> StateCard | None:
        stmt = select(StateCard).where(StateCard.session_id == session_id)
        return (await self.db.execute(stmt)).scalar_one_or_none()

    async def delete(self, session_id: UUID) -> bool:
        session = await self.get(session_id)
        if session is None:
            return False
        artifact_ids = list(
            (
                await self.db.execute(
                    select(Artifact.id).where(Artifact.session_id == session_id)
                )
            )
            .scalars()
            .all()
        )
        artifact_refs = list(
            (
                await self.db.execute(
                    select(ToolCall.result_ref).where(
                        ToolCall.session_id == session_id,
                        ToolCall.result_ref.is_not(None),
                    )
                )
            )
            .scalars()
            .all()
        )
        await self.db.execute(
            delete(Artifact).where(Artifact.session_id == session_id)
        )
        await self.db.execute(
            delete(Snapshot).where(Snapshot.session_id == session_id)
        )
        await self.db.execute(delete(ToolCall).where(ToolCall.session_id == session_id))
        await self.db.execute(delete(Message).where(Message.session_id == session_id))
        await self.db.execute(delete(Summary).where(Summary.session_id == session_id))
        await self.db.execute(delete(StateCard).where(StateCard.session_id == session_id))
        await self.db.execute(
            update(Todo).where(Todo.session_id == session_id).values(session_id=None)
        )
        await self.db.delete(session)
        await self.db.flush()
        if self.artifacts_dir is not None:
            ids = {str(item) for item in artifact_ids}
            ids.update(
                str(ref).removeprefix("artifact://")
                for ref in artifact_refs
                if ref and str(ref).startswith("artifact://")
            )
            for artifact_id in ids:
                if not artifact_id or "/" in artifact_id or "\\" in artifact_id:
                    continue
                for suffix in (".txt", ".meta.json"):
                    try:
                        (self.artifacts_dir / f"{artifact_id}{suffix}").unlink()
                    except OSError:
                        pass
        return True

    async def list_messages(self, session_id: UUID, limit: int = 500) -> list[Message]:
        stmt = (
            select(Message)
            .where(Message.session_id == session_id)
            .order_by(Message.turn_no.desc(), Message.created_at.desc())
            .limit(limit)
        )
        rows = list((await self.db.execute(stmt)).scalars().all())
        rows.reverse()
        return rows

    async def next_turn_no(self, session_id: UUID) -> int:
        stmt = select(func.coalesce(func.max(Message.turn_no), 0)).where(
            Message.session_id == session_id
        )
        value = (await self.db.execute(stmt)).scalar_one()
        return int(value) + 1

    async def add_message(
        self,
        *,
        session_id: UUID,
        turn_no: int,
        role: str,
        content: str,
        token_est: int = 0,
        meta: dict[str, object] | None = None,
    ) -> Message:
        message = Message(
            session_id=session_id,
            turn_no=turn_no,
            role=role,
            content=content,
            token_est=token_est,
            meta=meta or {},
        )
        self.db.add(message)
        await self.db.flush()
        return message

    async def add_tool_call(
        self,
        *,
        session_id: UUID,
        turn_no: int,
        tool_name: str,
        arguments: dict[str, object],
        status: str,
        result_summary: str | None,
        result_ref: str | None,
        error: str | None,
        latency_ms: int,
        idempotency_key: str | None,
        retry_count: int = 0,
        message_id: UUID | None = None,
        call_id: UUID | None = None,
    ) -> ToolCall:
        call = ToolCall(
            session_id=session_id,
            message_id=message_id,
            turn_no=turn_no,
            tool_name=tool_name,
            arguments=arguments,
            status=status,
            retry_count=retry_count,
            result_summary=result_summary,
            result_ref=result_ref,
            error=error,
            latency_ms=latency_ms,
            idempotency_key=idempotency_key,
        )
        if call_id is not None:
            call.id = call_id
        self.db.add(call)
        await self.db.flush()
        return call

    async def touch(self, session: Session) -> None:
        session.updated_at = func.now()
        await self.db.flush()

    async def count_task_turns(self, session_id: UUID) -> int:
        stmt = select(Session.task_turn_count).where(Session.id == session_id)
        return int((await self.db.execute(stmt)).scalar_one_or_none() or 0)

    async def list_tool_calls(self, session_id: UUID) -> list[ToolCall]:
        stmt = (
            select(ToolCall)
            .where(ToolCall.session_id == session_id)
            .order_by(ToolCall.created_at.asc())
        )
        return list((await self.db.execute(stmt)).scalars().all())


__all__ = ["SessionRepository"]

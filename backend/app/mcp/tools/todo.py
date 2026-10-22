from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.models.db_models import Todo, TodoOperation

_engine: Any = None
_sessionmaker: async_sessionmaker[AsyncSession] | None = None
_engine_url: str | None = None


class TodoError(ValueError):
    pass


def _db_url() -> str:
    import os

    url = os.environ.get("DATABASE_URL", "")
    if not url:
        raise TodoError("todo 工具缺少 DATABASE_URL 配置")
    return url


def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    global _engine, _sessionmaker, _engine_url
    url = _db_url()
    if _sessionmaker is None or _engine_url != url:
        _engine = create_async_engine(url, pool_pre_ping=True)
        _sessionmaker = async_sessionmaker(_engine, expire_on_commit=False)
        _engine_url = url
    return _sessionmaker


def _parse_due_at(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        due = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise TodoError("due_at 必须是 ISO 8601 时间字符串") from exc
    if due.tzinfo is None:
        due = due.replace(tzinfo=UTC)
    return due


def _serialize(todo: Todo) -> dict[str, Any]:
    return {
        "id": str(todo.id),
        "content": todo.content,
        "status": todo.status,
        "due_at": todo.due_at.isoformat() if todo.due_at else None,
        "client_op_id": todo.client_op_id,
        "session_id": str(todo.session_id) if todo.session_id else None,
        "created_at": todo.created_at.isoformat(),
        "updated_at": todo.updated_at.isoformat(),
    }


async def _find_by_op_id(db: AsyncSession, user_id: str, client_op_id: str) -> Todo | None:
    stmt = select(Todo).where(
        Todo.client_op_id == client_op_id, Todo.user_id == user_id
    )
    return (await db.execute(stmt)).scalar_one_or_none()


async def _replay_operation(
    db: AsyncSession, user_id: str, client_op_id: str | None, action: str
) -> dict[str, Any] | None:
    if not client_op_id:
        return None
    stmt = select(TodoOperation).where(
        TodoOperation.user_id == user_id,
        TodoOperation.client_op_id == client_op_id,
    )
    operation = (await db.execute(stmt)).scalar_one_or_none()
    if operation is None:
        return None
    if operation.action != action:
        raise TodoError(
            f"client_op_id 已用于 {operation.action} 操作，不能复用于 {action}"
        )
    return {**operation.result, "idempotent_replay": True}


async def _get_owned(db: AsyncSession, user_id: str, todo_id: uuid.UUID) -> Todo:
    stmt = select(Todo).where(
        Todo.id == todo_id, Todo.user_id == user_id, Todo.status != "deleted"
    )
    todo = (await db.execute(stmt)).scalar_one_or_none()
    if todo is None:
        raise TodoError("待办不存在或不属于当前用户")
    return todo


def _parse_uuid(value: Any) -> uuid.UUID:
    if not value or not isinstance(value, str):
        raise TodoError("id 不能为空且必须是 UUID 字符串")
    try:
        return uuid.UUID(value)
    except ValueError as exc:
        raise TodoError("id 必须是合法的 UUID") from exc


async def todo_tool(
    *,
    action: str,
    user_id: str | None = None,
    session_id: str | None = None,
    id: str | None = None,
    content: str | None = None,
    due_at: str | None = None,
    status: str | None = None,
    client_op_id: str | None = None,
) -> dict[str, Any]:
    if action not in {"create", "list", "update", "complete", "delete"}:
        raise TodoError("action 必须是 create/list/update/complete/delete")
    if not user_id:
        raise TodoError("缺少 user_id 上下文")

    sessionmaker = get_sessionmaker()
    async with sessionmaker() as db:
        if action == "list":
            stmt = (
                select(Todo)
                .where(Todo.user_id == user_id, Todo.status != "deleted")
                .order_by(Todo.created_at.desc())
            )
            if status:
                stmt = stmt.where(Todo.status == status)
            rows = list((await db.execute(stmt)).scalars().all())
            return {"todos": [_serialize(row) for row in rows], "count": len(rows)}

        replay = await _replay_operation(db, user_id, client_op_id, action)
        if replay is not None:
            return replay

        result: dict[str, Any]
        if action == "create":
            if not content or not content.strip():
                raise TodoError("create 操作 content 不能为空")
            op_id = client_op_id or uuid.uuid4().hex
            existing = await _find_by_op_id(db, user_id, op_id)
            if existing is not None:
                return {"todo": _serialize(existing), "idempotent_replay": True}
            todo = Todo(
                user_id=user_id,
                session_id=_maybe_uuid(session_id),
                content=content.strip(),
                status="open",
                due_at=_parse_due_at(due_at),
                client_op_id=op_id,
            )
            db.add(todo)
            try:
                await db.flush()
                result = {"todo": _serialize(todo)}
                db.add(
                    TodoOperation(
                        user_id=user_id,
                        client_op_id=op_id,
                        action="create",
                        todo_id=todo.id,
                        result=result,
                    )
                )
                await db.commit()
            except IntegrityError:
                await db.rollback()
                replay = await _replay_operation(db, user_id, op_id, "create")
                if replay is not None:
                    return replay
                existing = await _find_by_op_id(db, user_id, op_id)
                if existing is None:
                    raise
                return {"todo": _serialize(existing), "idempotent_replay": True}
            await db.refresh(todo)
            return {"todo": _serialize(todo)}

        if action == "delete":
            todo = await _get_owned(db, user_id, _parse_uuid(id))
            todo.status = "deleted"
            result = {"deleted": True, "id": str(todo.id)}
        else:
            todo = await _get_owned(db, user_id, _parse_uuid(id))
            if action == "complete":
                todo.status = "done"
            elif action == "update":
                if content is not None:
                    if not content.strip():
                        raise TodoError("content 不能为空")
                    todo.content = content.strip()
                if status is not None:
                    if status not in {"open", "done"}:
                        raise TodoError("status 只能是 open 或 done")
                    todo.status = status
                if due_at is not None:
                    todo.due_at = _parse_due_at(due_at)
            result = {"todo": _serialize(todo)}

        if client_op_id:
            db.add(
                TodoOperation(
                    user_id=user_id,
                    client_op_id=client_op_id,
                    action=action,
                    todo_id=todo.id,
                    result=result,
                )
            )
        try:
            await db.commit()
        except IntegrityError:
            await db.rollback()
            replay = await _replay_operation(db, user_id, client_op_id, action)
            if replay is not None:
                return replay
            raise
        return result


def _maybe_uuid(value: str | None) -> uuid.UUID | None:
    if not value:
        return None
    try:
        return uuid.UUID(value)
    except ValueError:
        return None


__all__ = ["TodoError", "get_sessionmaker", "todo_tool"]

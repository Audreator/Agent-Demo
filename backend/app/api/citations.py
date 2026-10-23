from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.citations import LinkStatus, check_links, public_citation
from app.config import Settings
from app.db import get_db
from app.models.db_models import Session, ToolCall
from app.observability.tracing import get_tracer
from app.sessions.schemas import CitationCheckIn, CitationCheckOut, CitationOut

router = APIRouter()
DbSession = Annotated[AsyncSession, Depends(get_db)]


def _settings(request: Request) -> Settings:
    return request.app.state.app_state.settings


@router.post("/citations/check", response_model=CitationCheckOut)
async def check_citations(
    body: CitationCheckIn, db: DbSession, request: Request
) -> CitationCheckOut:
    state = request.app.state.app_state
    settings: Settings = state.settings
    session = await db.get(Session, body.session_id)
    if session is None or session.user_id != settings.dev_user_id:
        raise HTTPException(status_code=404, detail="会话不存在")

    stmt = (
        select(ToolCall.session_id, ToolCall.turn_no, ToolCall.result_meta)
        .join(Session, Session.id == ToolCall.session_id)
        .where(
            Session.user_id == settings.dev_user_id,
            ToolCall.tool_name == "search",
            ToolCall.session_id == body.session_id,
        )
    )
    if body.turn_no is not None:
        stmt = stmt.where(ToolCall.turn_no == body.turn_no)

    stored: dict[str, dict[str, Any]] = {}
    for _session_id, _turn_no, result_meta in (await db.execute(stmt)).all():
        citations = (result_meta or {}).get("citations")
        if not isinstance(citations, list):
            continue
        for item in citations:
            if not isinstance(item, dict):
                continue
            citation_id = str(item.get("id") or "")
            if citation_id and citation_id not in stored:
                stored[citation_id] = item

    requested: list[dict[str, Any]] = []
    missing: list[str] = []
    seen: set[str] = set()
    for citation_id in body.citation_ids:
        if citation_id in seen:
            continue
        seen.add(citation_id)
        item = stored.get(citation_id)
        if item is None:
            missing.append(citation_id)
            continue
        requested.append(item)
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"包含非当前会话已存引用: {', '.join(missing[:3])}",
        )
    if len(requested) > settings.citation_max_total:
        raise HTTPException(status_code=400, detail="单次最多校验 10 条引用")

    statuses: dict[str, LinkStatus] = {}
    stale_urls: list[str] = []
    ttl = timedelta(seconds=settings.link_check_cache_ttl)
    now = datetime.now(UTC)
    for item in requested:
        url = str(item.get("url") or "")
        checked_at = item.get("checked_at")
        if url and checked_at:
            try:
                checked = datetime.fromisoformat(str(checked_at))
            except ValueError:
                checked = None
            if checked is not None:
                if checked.tzinfo is None:
                    checked = checked.replace(tzinfo=UTC)
                if now - checked < ttl:
                    statuses[url] = LinkStatus(
                        url,
                        bool(item.get("reachable")),
                        item.get("status_code")
                        if isinstance(item.get("status_code"), int)
                        else None,
                        str(item.get("reason") or ""),
                        bool(item.get("rejected")),
                        checked,
                    )
                    continue
        stale_urls.append(url)
    if stale_urls:
        statuses.update(
            await check_links(
                stale_urls,
                timeout=settings.link_check_timeout,
                concurrency=settings.link_check_concurrency,
                cache=state.link_cache,
            )
        )
    results: list[CitationOut] = []
    tracer = get_tracer(settings)
    for item in requested:
        status = statuses.get(str(item.get("url") or ""))
        payload = dict(item)
        if status is not None:
            payload.update(
                {
                    "reachable": status.reachable,
                    "status_code": status.status_code,
                    "checked_at": status.checked_at.isoformat(),
                    "reason": status.reason,
                }
            )
            if status.rejected:
                await tracer.emit_event(
                    "link_check_rejected",
                    {
                        "url": status.url,
                        "reason": status.reason,
                        "source": "citation_check",
                    },
                    session_id=str(body.session_id),
                    turn_no=body.turn_no,
                )
        results.append(CitationOut(**public_citation(payload)))
    return CitationCheckOut(citations=results)


__all__ = ["router"]

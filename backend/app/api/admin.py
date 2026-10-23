from __future__ import annotations

import csv
import io
from datetime import datetime
from typing import Annotated, Any
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.admin.service import compute_metrics, list_turns, parse_range, turn_detail
from app.config import Settings
from app.db import get_db

TZ = ZoneInfo("Asia/Shanghai")

router = APIRouter(prefix="/admin")
DbSession = Annotated[AsyncSession, Depends(get_db)]

FORMULA_PREFIXES = ("=", "+", "-", "@")


def _settings(request: Request) -> Settings:
    return request.app.state.app_state.settings


def _range(
    settings: Settings, start: str | None, end: str | None
) -> tuple[datetime, datetime]:
    try:
        return parse_range(settings, start, end)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/metrics")
async def metrics(
    db: DbSession,
    request: Request,
    start: str | None = None,
    end: str | None = None,
) -> dict[str, Any]:
    settings = _settings(request)
    range_start, range_end = _range(settings, start, end)
    return await compute_metrics(db, settings, range_start, range_end)


@router.get("/turns")
async def turns(
    db: DbSession,
    request: Request,
    start: str | None = None,
    end: str | None = None,
    page: int = 1,
    page_size: int | None = None,
    session_id: str | None = None,
    tool_name: str | None = None,
    tool_status: str | None = None,
    rating: str | None = None,
    finish_reason: str | None = None,
    trace_id: str | None = None,
) -> dict[str, Any]:
    settings = _settings(request)
    range_start, range_end = _range(settings, start, end)
    size = settings.admin_page_size if page_size is None else page_size
    if size < 1 or size > 200:
        raise HTTPException(status_code=400, detail="page_size 必须在 1-200 之间")
    if page < 1:
        raise HTTPException(status_code=400, detail="page 必须大于 0")
    return await list_turns(
        db,
        settings,
        range_start,
        range_end,
        page=page,
        page_size=size,
        session_id=session_id,
        tool_name=tool_name,
        tool_status=tool_status,
        rating=rating,
        finish_reason=finish_reason,
        trace_id=trace_id,
    )


@router.get("/turns/{trace_id}")
async def turns_detail(
    trace_id: str, db: DbSession, request: Request
) -> dict[str, Any]:
    settings = _settings(request)
    detail = await turn_detail(db, settings, trace_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="调用链不存在")
    return detail


@router.get("/export")
async def export_turns(
    db: DbSession,
    request: Request,
    start: str | None = None,
    end: str | None = None,
    session_id: str | None = None,
    tool_name: str | None = None,
    tool_status: str | None = None,
    rating: str | None = None,
    finish_reason: str | None = None,
    trace_id: str | None = None,
) -> StreamingResponse:
    settings = _settings(request)
    range_start, range_end = _range(settings, start, end)
    filters = {
        "session_id": session_id,
        "tool_name": tool_name,
        "tool_status": tool_status,
        "rating": rating,
        "finish_reason": finish_reason,
        "trace_id": trace_id,
    }
    total = (
        await list_turns(
            db, settings, range_start, range_end, page=1, page_size=1, **filters
        )
    )["total"]
    if total > settings.admin_export_max_rows:
        raise HTTPException(
            status_code=400,
            detail=f"结果超过 {settings.admin_export_max_rows} 行，请缩小时间范围或增加筛选条件",
        )
    rows = (
        await list_turns(
            db,
            settings,
            range_start,
            range_end,
            page=1,
            page_size=max(1, total),
            **filters,
        )
    )["items"]
    content = _build_csv(rows)
    filename = (
        f"agent-turns-{range_start.astimezone(TZ).strftime('%Y%m%d-%H%M')}-"
        f"{range_end.astimezone(TZ).strftime('%Y%m%d-%H%M')}.csv"
    )
    return StreamingResponse(
        io.BytesIO(content),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def _safe_cell(value: Any) -> str:
    text = "" if value is None else str(value)
    if text.startswith(FORMULA_PREFIXES):
        return "'" + text
    return text


def _local_time(value: Any) -> str:
    if not value:
        return ""
    try:
        return datetime.fromisoformat(str(value)).astimezone(TZ).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    except ValueError:
        return str(value)


def _build_csv(rows: list[dict[str, Any]]) -> bytes:
    fieldnames = [
        "时间",
        "session_id",
        "turn_no",
        "trace_id",
        "用户输入",
        "最终回答",
        "模型层",
        "工具链",
        "工具状态",
        "轮次耗时ms",
        "token",
        "评价",
        "原因",
    ]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        tools = row.get("tools") or []
        chain = "; ".join(
            f"{tool.get('name')}:{tool.get('status')}:{tool.get('latency_ms')}ms"
            for tool in tools
        )
        statuses = ", ".join(sorted({str(tool.get("status")) for tool in tools}))
        feedback = row.get("feedback") or {}
        reason = ""
        if feedback:
            reason = str(feedback.get("reason_code") or "")
            if feedback.get("reason_text"):
                reason += f": {feedback['reason_text']}"
        writer.writerow(
            {
                "时间": _safe_cell(_local_time(row.get("started_at"))),
                "session_id": _safe_cell(row.get("session_id")),
                "turn_no": _safe_cell(row.get("turn_no")),
                "trace_id": _safe_cell(row.get("trace_id")),
                "用户输入": _safe_cell(row.get("user_input")),
                "最终回答": _safe_cell(row.get("final_answer")),
                "模型层": _safe_cell(row.get("model_layer")),
                "工具链": _safe_cell(chain),
                "工具状态": _safe_cell(statuses),
                "轮次耗时ms": _safe_cell(row.get("duration_ms")),
                "token": _safe_cell(row.get("tokens")),
                "评价": _safe_cell(feedback.get("rating")),
                "原因": _safe_cell(reason),
            }
        )
    return ("\ufeff" + buffer.getvalue()).encode("utf-8")


__all__ = ["router"]

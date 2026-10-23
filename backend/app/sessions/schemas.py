from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class FeedbackSummaryOut(BaseModel):
    rating: str | None = None
    reason_code: str | None = None
    reason_text: str | None = None
    likes: int = 0
    dislikes: int = 0
    updated_at: datetime | None = None


class SessionCreate(BaseModel):
    title: str | None = Field(default=None, max_length=200)


class SessionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    user_id: str
    title: str
    mode: str
    task_turn_count: int
    status: str
    last_trace_id: str | None = None
    created_at: datetime
    updated_at: datetime


class MessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    session_id: UUID
    turn_no: int
    role: str
    content: str
    token_est: int
    meta: dict[str, Any]
    created_at: datetime
    feedback: FeedbackSummaryOut | None = None


class SendMessageRequest(BaseModel):
    content: str = Field(min_length=1, max_length=100_000)
    last_event_id: int | None = None
    client_message_id: str | None = Field(default=None, max_length=128)


class ToolInfo(BaseModel):
    name: str
    description: str
    inputSchema: dict[str, Any]
    online: bool = False


class ToolCallOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    session_id: UUID
    turn_no: int
    tool_name: str
    arguments: dict[str, Any]
    status: str
    retry_count: int
    latency_ms: int
    result_summary: str | None = None
    result_ref: str | None = None
    result_meta: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    created_at: datetime


class ToolResultOut(BaseModel):
    tool_call_id: UUID
    content: str
    source: str
    result_ref: str | None = None


class FeedbackIn(BaseModel):
    rating: str
    reason_code: str | None = None
    reason_text: str | None = None


class FeedbackOut(BaseModel):
    message_id: UUID
    rating: str
    reason_code: str | None = None
    reason_text: str | None = None
    likes: int = 0
    dislikes: int = 0
    updated_at: datetime


class CitationOut(BaseModel):
    id: str
    title: str
    url: str
    snippet: str = ""
    reachable: bool
    status_code: int | None = None
    checked_at: str | None = None
    reason: str = ""


class CitationCheckIn(BaseModel):
    session_id: UUID
    turn_no: int | None = None
    citation_ids: list[str] = Field(min_length=1, max_length=10)


class CitationCheckOut(BaseModel):
    citations: list[CitationOut]


class TraceEvent(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    trace_id: str
    span_id: str
    parent_span_id: str | None
    session_id: str | None
    turn_no: int | None
    kind: str
    payload: dict[str, Any]
    duration_ms: int | None
    ts: datetime


class HealthOut(BaseModel):
    status: str
    models: dict[str, Any]
    mcp: dict[str, Any]

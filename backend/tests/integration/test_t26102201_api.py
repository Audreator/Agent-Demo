from __future__ import annotations

import re
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx

from app.admin.service import compute_alerts
from app.config import Settings
from app.main import create_app
from app.models.db_models import Event, Message, MessageFeedback, Session, ToolCall
from app.models.gateway import ModelGateway
from app.observability.tracing import get_tracer
from tests.helpers import FakeMCP, ScriptedProvider


def _provider(*scripts: Any) -> ScriptedProvider:
    return ScriptedProvider(scripts=list(scripts))


def _search_payload(url: str = "https://example.com/page") -> dict[str, Any]:
    citation = {
        "id": f"tc-{abs(hash(url))}-c1",
        "title": "示例标题",
        "url": url,
        "snippet": "示例摘要",
        "reachable": url.startswith("https://"),
        "status_code": 200 if url.startswith("https://") else None,
        "checked_at": "2020-01-01T00:00:00+00:00",
    }
    return {
        "query": "测试",
        "source": "deepseek_web_search",
        "count": 1,
        "results": [{"title": citation["title"], "url": url, "snippet": citation["snippet"]}],
        "citations": [citation],
        "reachable_count": 1 if citation["reachable"] else 0,
    }


async def _stream_turn(client: httpx.AsyncClient, session_id: str, content: str) -> str:
    async with client.stream(
        "POST", f"/api/sessions/{session_id}/messages", json={"content": content}
    ) as response:
        assert response.status_code == 200
        return (await response.aread()).decode("utf-8")


async def _final_message(client: httpx.AsyncClient, session_id: str) -> dict[str, Any]:
    messages = (await client.get(f"/api/sessions/{session_id}/messages")).json()
    finals = [
        item
        for item in messages
        if item["role"] == "assistant" and item["meta"].get("kind") in {"final", "wrap_up"}
    ]
    assert finals
    return finals[-1]


async def test_search_citations_tool_calls_and_feedback(settings: Settings) -> None:
    provider = _provider(
        'Thought: 搜索\nAction: search\nAction Input: {"query": "测试"}',
        "Thought: 汇总\nFinal Answer: 已找到资料。",
    )
    mcp = FakeMCP(responses={"search": _search_payload()})
    gateway = ModelGateway(settings, providers=[provider], tracer=get_tracer(settings))
    app = create_app(settings, gateway=gateway, mcp=mcp, auto_start_mcp=True)

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            created = await client.post("/api/sessions", json={"title": "引用测试"})
            session_id = created.json()["id"]
            text = await _stream_turn(client, session_id, "请搜索测试资料")
            assert "event: tool_result" in text
            assert "citations" in text
            assert "示例标题" in text

            calls = (
                await client.get(f"/api/sessions/{session_id}/tool-calls")
            ).json()
            assert len(calls) == 1
            call = calls[0]
            assert call["tool_name"] == "search"
            assert call["status"] == "ok"
            citations = call["result_meta"]["citations"]
            assert citations and citations[0]["reachable"] is True
            assert citations[0]["status_code"] == 200

            result = await client.get(
                f"/api/sessions/{session_id}/tool-calls/{call['id']}/result"
            )
            assert result.status_code == 200
            assert "示例标题" in result.json()["content"]

            other_session = (await client.post("/api/sessions", json={})).json()["id"]
            cross = await client.get(
                f"/api/sessions/{other_session}/tool-calls/{call['id']}/result"
            )
            assert cross.status_code == 404

            final = await _final_message(client, session_id)
            assert final["feedback"]["rating"] is None
            feedback_url = (
                f"/api/sessions/{session_id}/messages/{final['id']}/feedback"
            )
            liked = await client.post(feedback_url, json={"rating": "like"})
            assert liked.status_code == 200
            assert liked.json()["likes"] == 1

            updated = await client.post(
                feedback_url,
                json={"rating": "dislike", "reason_code": "too_slow"},
            )
            assert updated.status_code == 200
            assert updated.json()["rating"] == "dislike"
            assert updated.json()["likes"] == 0
            assert updated.json()["dislikes"] == 1

            messages = (await client.get(f"/api/sessions/{session_id}/messages")).json()
            final_row = next(item for item in messages if item["id"] == final["id"])
            assert final_row["feedback"]["rating"] == "dislike"
            assert final_row["feedback"]["reason_code"] == "too_slow"

            invalid_other = await client.post(
                feedback_url,
                json={"rating": "dislike", "reason_code": "other", "reason_text": " "},
            )
            assert invalid_other.status_code == 422
            invalid_reason = await client.post(
                feedback_url, json={"rating": "dislike", "reason_code": "nope"}
            )
            assert invalid_reason.status_code == 422
            other_filled = await client.post(
                feedback_url,
                json={
                    "rating": "dislike",
                    "reason_code": "other",
                    "reason_text": "回答缺少具体来源",
                },
            )
            assert other_filled.status_code == 200
            assert other_filled.json()["reason_text"] == "回答缺少具体来源"
            like_with_reason = await client.post(
                feedback_url, json={"rating": "like", "reason_code": "too_slow"}
            )
            assert like_with_reason.status_code == 422

            cancelled = await client.delete(feedback_url)
            assert cancelled.status_code == 204
            messages = (await client.get(f"/api/sessions/{session_id}/messages")).json()
            final_row = next(item for item in messages if item["id"] == final["id"])
            assert final_row["feedback"]["rating"] is None
            assert final_row["feedback"]["likes"] == 0


async def test_citations_check_rejects_unknown_and_private(settings: Settings) -> None:
    provider = _provider(
        'Thought: 搜索\nAction: search\nAction Input: {"query": "内网"}',
        "Thought: 完成\nFinal Answer: 已检查。",
    )
    mcp = FakeMCP(responses={"search": _search_payload("http://127.0.0.1:9/private")})
    gateway = ModelGateway(settings, providers=[provider], tracer=get_tracer(settings))
    app = create_app(settings, gateway=gateway, mcp=mcp, auto_start_mcp=True)

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            session_id = (await client.post("/api/sessions", json={})).json()["id"]
            await _stream_turn(client, session_id, "搜索内网资料")
            calls = (
                await client.get(f"/api/sessions/{session_id}/tool-calls")
            ).json()
            assert calls[0]["status"] == "ok"
            citation_id = calls[0]["result_meta"]["citations"][0]["id"]

            checked = await client.post(
                "/api/citations/check",
                json={"session_id": session_id, "citation_ids": [citation_id]},
            )
            assert checked.status_code == 200
            item = checked.json()["citations"][0]
            assert item["reachable"] is False
            assert item["reason"]

            without_session = await client.post(
                "/api/citations/check", json={"citation_ids": [citation_id]}
            )
            assert without_session.status_code == 422

            correct_turn = await client.post(
                "/api/citations/check",
                json={
                    "session_id": session_id,
                    "turn_no": calls[0]["turn_no"],
                    "citation_ids": [citation_id],
                },
            )
            assert correct_turn.status_code == 200
            wrong_turn = await client.post(
                "/api/citations/check",
                json={
                    "session_id": session_id,
                    "turn_no": calls[0]["turn_no"] + 10,
                    "citation_ids": [citation_id],
                },
            )
            assert wrong_turn.status_code == 400

            unknown = await client.post(
                "/api/citations/check",
                json={"session_id": session_id, "citation_ids": ["missing-id"]},
            )
            assert unknown.status_code == 400

            other_session = (await client.post("/api/sessions", json={})).json()["id"]
            wrong_session = await client.post(
                "/api/citations/check",
                json={"session_id": other_session, "citation_ids": [citation_id]},
            )
            assert wrong_session.status_code == 400

            too_many = await client.post(
                "/api/citations/check",
                json={"session_id": session_id, "citation_ids": [citation_id] * 11},
            )
            assert too_many.status_code == 422


async def test_tool_result_falls_back_for_legacy_tool_call(settings: Settings, db: Any) -> None:
    session = Session(user_id=settings.dev_user_id, title="旧数据")
    db.add(session)
    await db.flush()
    call = ToolCall(
        session_id=session.id,
        turn_no=7,
        tool_name="calculator",
        arguments={},
        status="ok",
        result_summary="旧结果",
        result_ref=None,
        error=None,
        latency_ms=10,
        idempotency_key=None,
    )
    db.add(call)
    db.add(
        Message(
            session_id=session.id,
            turn_no=7,
            role="tool",
            content="旧版完整结果正文",
            token_est=8,
            meta={"tool_call_id": "legacy-hex-id", "tool_name": "calculator"},
        )
    )
    await db.commit()

    gateway = ModelGateway(settings, providers=[_provider()], tracer=get_tracer(settings))
    app = create_app(settings, gateway=gateway, mcp=FakeMCP(), auto_start_mcp=True)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.get(
                f"/api/sessions/{session.id}/tool-calls/{call.id}/result"
            )
            assert response.status_code == 200
            assert response.json()["content"] == "旧版完整结果正文"
            assert response.json()["source"] == "message"


async def test_admin_alerts_thresholds(settings: Settings, db: Any) -> None:
    session = Session(user_id=settings.dev_user_id, title="告警测试")
    db.add(session)
    await db.flush()

    for index in range(10):
        db.add(
            ToolCall(
                session_id=session.id,
                turn_no=1,
                tool_name="weather",
                arguments={"location": "杭州"},
                status="error",
                result_summary=None,
                result_ref=None,
                error="upstream timeout",
                latency_ms=100,
                idempotency_key=None,
            )
        )
        trace_id = uuid.uuid4().hex
        db.add(
            Event(
                trace_id=trace_id,
                span_id=uuid.uuid4().hex[:16],
                parent_span_id=None,
                session_id=str(session.id),
                turn_no=index + 1,
                kind="turn",
                payload={},
                duration_ms=120_000,
            )
        )
    db.add(
        Event(
            trace_id=uuid.uuid4().hex,
            span_id=uuid.uuid4().hex[:16],
            parent_span_id=None,
            session_id=str(session.id),
            turn_no=1,
            kind="breaker_open",
            payload={"provider": "deepseek"},
            duration_ms=0,
        )
    )
    for index in range(5):
        message = Message(
            session_id=session.id,
            turn_no=20 + index,
            role="assistant",
            content="最终回答",
            token_est=4,
            meta={"kind": "final"},
        )
        db.add(message)
        await db.flush()
        db.add(
            MessageFeedback(
                message_id=message.id,
                session_id=session.id,
                user_id=settings.dev_user_id,
                rating="like" if index == 0 else "dislike",
                reason_code="too_slow",
            )
        )
    for index in range(3):
        db.add(
            Message(
                session_id=session.id,
                turn_no=40 + index,
                role="assistant",
                content="预算收束",
                token_est=4,
                meta={"kind": "budget_exceeded"},
            )
        )
    await db.commit()

    alerts = await compute_alerts(db, settings)
    rules = {item["rule"] for item in alerts}
    assert {
        "tool_failure_rate",
        "satisfaction_rate",
        "turn_p95",
        "breaker_open",
        "wrap_up",
    } <= rules


async def test_admin_metrics_turns_detail_and_export(settings: Settings) -> None:
    provider = _provider(
        'Thought: 计算\nAction: calculator\nAction Input: {"expression": "1+1"}',
        {
            "text": "Thought: 完成\nFinal Answer: 结果是 2。",
            "usage": {"total_tokens": 42},
        },
    )
    mcp = FakeMCP(
        responses={
            "calculator": {"expression": "1+1", "result": "2", "formatted": "2"}
        }
    )
    gateway = ModelGateway(settings, providers=[provider], tracer=get_tracer(settings))
    app = create_app(settings, gateway=gateway, mcp=mcp, auto_start_mcp=True)

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            session_id = (await client.post("/api/sessions", json={})).json()["id"]
            text = await _stream_turn(client, session_id, "=1+1 请计算")
            assert "event: done" in text
            trace_match = re.search(r'"trace_id": "([a-f0-9]+)"', text)
            assert trace_match is not None

            final = await _final_message(client, session_id)
            liked = await client.post(
                f"/api/sessions/{session_id}/messages/{final['id']}/feedback",
                json={"rating": "like"},
            )
            assert liked.status_code == 200

            now = datetime.now(UTC)
            start = (now - timedelta(minutes=10)).isoformat()
            end = (now + timedelta(minutes=2)).isoformat()
            metrics = await client.get(
                "/api/admin/metrics", params={"start": start, "end": end}
            )
            assert metrics.status_code == 200
            payload = metrics.json()
            assert payload["overview"]["turns"] == 1
            assert payload["overview"]["task_turns"] == 1
            assert payload["overview"]["tool_calls"] == 1
            assert payload["overview"]["tool_success_rate"] == 1.0
            assert payload["overview"]["satisfaction_rate"] == 1.0
            assert payload["overview"]["total_tokens"] == 42
            assert payload["tools"][0]["tool_name"] == "calculator"
            assert any(item["layer"] == "deepseek" for item in payload["models"])
            assert payload["satisfaction"]["total"] == 1

            calls = (
                await client.get(f"/api/sessions/{session_id}/tool-calls")
            ).json()
            assert calls[0]["result_meta"] == {}
            assert calls[0]["retry_count"] == 0

            turns = await client.get(
                "/api/admin/turns",
                params={"start": start, "end": end, "page": 1, "page_size": 50},
            )
            assert turns.status_code == 200
            turn_items = turns.json()["items"]
            assert len(turn_items) == 1
            assert turn_items[0]["tools"][0]["name"] == "calculator"
            trace_id = turn_items[0]["trace_id"]

            detail = await client.get(f"/api/admin/turns/{trace_id}")
            assert detail.status_code == 200
            detail_payload = detail.json()
            kinds = {event["kind"] for event in detail_payload["events"]}
            assert "turn" in kinds
            assert "model_call" in kinds
            assert "tool_call" in kinds
            assert detail_payload["messages"]

            filtered = await client.get(
                "/api/admin/turns",
                params={
                    "start": start,
                    "end": end,
                    "tool_name": "calculator",
                    "tool_status": "ok",
                    "rating": "like",
                    "finish_reason": "final_answer",
                },
            )
            assert filtered.status_code == 200
            assert filtered.json()["total"] == 1

            missing_turn = await client.get("/api/admin/turns/not-found")
            assert missing_turn.status_code == 404

            exported = await client.get(
                "/api/admin/export", params={"start": start, "end": end}
            )
            assert exported.status_code == 200
            assert exported.content.startswith(b"\xef\xbb\xbf")
            csv_text = exported.content.decode("utf-8-sig")
            assert "calculator:ok" in csv_text
            assert "'=1+1 请计算" in csv_text

            too_long = await client.get(
                "/api/admin/metrics",
                params={
                    "start": (now - timedelta(days=120)).isoformat(),
                    "end": now.isoformat(),
                },
            )
            assert too_long.status_code == 400
            invalid = await client.get(
                "/api/admin/metrics",
                params={"start": end, "end": start},
            )
            assert invalid.status_code == 400

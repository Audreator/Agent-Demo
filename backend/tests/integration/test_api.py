from __future__ import annotations

import re

import httpx

from app.config import Settings
from app.main import create_app
from app.models.gateway import ModelGateway
from app.observability.tracing import get_tracer
from tests.helpers import EchoProvider, FakeMCP


async def test_api_contract(settings: Settings) -> None:
    provider = EchoProvider()
    gateway = ModelGateway(settings, providers=[provider], tracer=get_tracer(settings))
    mcp = FakeMCP()
    app = create_app(settings, gateway=gateway, mcp=mcp, auto_start_mcp=True)

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            health = await client.get("/api/health")
            assert health.status_code == 200
            assert health.json()["status"] == "ok"

            tools = await client.get("/api/tools")
            assert tools.status_code == 200
            tool_names = {tool["name"] for tool in tools.json()}
            assert tool_names == {"calculator", "todo", "search", "weather", "read_docs"}
            for tool in tools.json():
                assert tool["inputSchema"]["type"] == "object"
                assert tool["online"] is True

            created = await client.post("/api/sessions", json={"title": "API 会话"})
            assert created.status_code == 201
            session_id = created.json()["id"]

            async with client.stream(
                "POST",
                f"/api/sessions/{session_id}/messages",
                json={"content": "这里的回答是什么？"},
            ) as response:
                assert response.status_code == 200
                assert response.headers["content-type"].startswith("text/event-stream")
                body = await response.aread()
            text = body.decode("utf-8")
            assert "event: meta" in text
            assert "event: done" in text
            assert "finish_reason" in text
            trace_match = re.search(r'"trace_id": "([a-f0-9]+)"', text)
            assert trace_match is not None
            trace_id = trace_match.group(1)

            messages = await client.get(f"/api/sessions/{session_id}/messages")
            assert messages.status_code == 200
            payload = messages.json()
            assert payload[0]["role"] == "user"
            assert any(item["role"] == "assistant" for item in payload)

            async with client.stream(
                "POST",
                f"/api/sessions/{session_id}/messages",
                json={"content": "不应执行的新消息", "last_event_id": 1},
            ) as replay:
                replay_body = await replay.aread()
            replay_text = replay_body.decode("utf-8")
            assert "event: done" in replay_text
            assert "event: meta" not in replay_text
            after_replay = await client.get(f"/api/sessions/{session_id}/messages")
            assert len(after_replay.json()) == len(payload)

            seq_ids = [int(value) for value in re.findall(r"^id: (\d+)$", text, re.M)]
            async with client.stream(
                "POST",
                f"/api/sessions/{session_id}/messages",
                json={"content": "不应执行", "last_event_id": max(seq_ids)},
            ) as tail:
                tail_text = (await tail.aread()).decode("utf-8")
            assert "already_completed" in tail_text

            first_retry_body = {
                "content": "同一条幂等消息",
                "client_message_id": "cid-1",
            }
            async with client.stream(
                "POST", f"/api/sessions/{session_id}/messages", json=first_retry_body
            ) as first_stream:
                await first_stream.aread()
            after_first = (await client.get(f"/api/sessions/{session_id}/messages")).json()
            async with client.stream(
                "POST", f"/api/sessions/{session_id}/messages", json=first_retry_body
            ) as duplicate:
                duplicate_text = (await duplicate.aread()).decode("utf-8")
            after_duplicate = (
                await client.get(f"/api/sessions/{session_id}/messages")
            ).json()
            assert "event: done" in duplicate_text
            assert len(after_duplicate) == len(after_first)

            async with client.stream(
                "POST",
                f"/api/sessions/{session_id}/messages",
                json={"content": "最后一轮", "client_message_id": "cid-2", "last_event_id": 0},
            ) as new_turn:
                new_turn_text = (await new_turn.aread()).decode("utf-8")
            assert "event: done" in new_turn_text
            after_new = (await client.get(f"/api/sessions/{session_id}/messages")).json()
            assert len(after_new) > len(after_duplicate)

            trace = await client.get(f"/api/traces/{trace_id}")
            assert trace.status_code == 200
            events = trace.json()
            assert events
            assert all(event["trace_id"] == trace_id for event in events)
            kinds = {event["kind"] for event in events}
            assert "turn" in kinds
            assert "context_build" in kinds

            sessions = await client.get("/api/sessions")
            assert sessions.status_code == 200
            assert any(item["id"] == session_id for item in sessions.json())

            deleted = await client.delete(f"/api/sessions/{session_id}")
            assert deleted.status_code == 204
            missing = await client.get(f"/api/sessions/{session_id}/messages")
            assert missing.status_code == 404


async def test_tools_offline_when_mcp_stopped(settings: Settings) -> None:
    gateway = ModelGateway(settings, providers=[EchoProvider()], tracer=get_tracer(settings))
    app = create_app(
        settings,
        gateway=gateway,
        mcp=FakeMCP(started=False),
        auto_start_mcp=False,
    )
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            response = await client.get("/api/tools")
            assert response.status_code == 200
            tools = response.json()
            assert len(tools) == 5
            assert all(tool["online"] is False for tool in tools)


async def test_send_to_unknown_session_returns_404(settings: Settings) -> None:
    gateway = ModelGateway(settings, providers=[EchoProvider()], tracer=get_tracer(settings))
    app = create_app(settings, gateway=gateway, mcp=FakeMCP(), auto_start_mcp=True)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            response = await client.post(
                "/api/sessions/00000000-0000-0000-0000-000000000001/messages",
                json={"content": "hi"},
            )
            assert response.status_code == 404

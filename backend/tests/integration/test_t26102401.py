from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
from sqlalchemy import select

from app.admin.service import compute_metrics
from app.agent.uncertainty import FORCED_SEARCH_EMPTY_TEXT
from app.config import Settings
from app.main import create_app
from app.models.db_models import Event, Message, Session, ToolCall
from app.models.gateway import ModelGateway
from app.observability.tracing import get_tracer
from tests.helpers import Collector, FakeMCP, ScriptedProvider, registry
from tests.integration.test_loop import build_loop, create_session


def _search_payload(
    *, url: str = "https://example.com/page", empty: bool = False
) -> dict[str, Any]:
    if empty:
        return {
            "query": "测试",
            "source": "deepseek_web_search",
            "count": 0,
            "results": [],
            "citations": [],
            "reachable_count": 0,
        }
    citation = {
        "id": f"tc-{abs(hash(url))}-c1",
        "title": "火星人口白皮书",
        "url": url,
        "snippet": "2026 年火星常住人口为 42 人",
        "reachable": url.startswith("https://"),
        "status_code": 200 if url.startswith("https://") else None,
        "checked_at": "2020-01-01T00:00:00+00:00",
    }
    return {
        "query": "测试",
        "source": "deepseek_web_search",
        "count": 1,
        "results": [
            {"title": citation["title"], "url": url, "snippet": citation["snippet"]}
        ],
        "citations": [citation],
        "reachable_count": 1 if citation["reachable"] else 0,
    }


def _deltas(collector: Collector) -> str:
    return "".join(
        str(data.get("delta", ""))
        for name, data in collector.events
        if name == "message_delta"
    )


async def _events(db: Any, session_id: Any, *kinds: str) -> list[Event]:
    rows = (
        await db.execute(
            select(Event)
            .where(Event.session_id == str(session_id), Event.kind.in_(kinds))
            .order_by(Event.ts.asc(), Event.id.asc())
        )
    ).scalars().all()
    return list(rows)


async def test_refusal_triggers_forced_search_and_answers(settings: Settings, db: Any) -> None:
    session = await create_session(db, settings, "强制检索")
    provider = ScriptedProvider(
        scripts=[
            "Thought: 没有把握\nFinal Answer: 我不知道。",
            "Thought: 已检索\nFinal Answer: 根据检索，2026 年火星常住人口为 42 人。",
        ]
    )
    mcp = FakeMCP(responses={"search": _search_payload()})
    loop, _ = build_loop(settings, provider, mcp)
    collector = Collector()
    user_input = "2026 年火星常住人口是多少？"

    result = await loop.run(session.id, user_input, collector)

    assert result.finish_reason == "final_answer"
    assert "42" in result.final_answer
    deltas = [
        data["delta"] for name, data in collector.events if name == "message_delta"
    ]
    assert len(deltas) == 1
    assert "42" in deltas[0]
    assert "我不知道" not in _deltas(collector)
    names = collector.names()
    assert names.count("tool_start") == 1
    tool_result = collector.first("tool_result")
    assert tool_result is not None
    assert tool_result["status"] == "ok"
    assert tool_result["citations"]

    assert len(mcp.calls) == 1
    _, arguments, _ = mcp.calls[0]
    assert arguments["query"] == user_input
    assert arguments["max_results"] == 5
    second_prompt = "\n".join(
        str(message.get("content")) for message in provider.calls[1]
    )
    assert "Observation:" in second_prompt
    assert "火星人口白皮书" in second_prompt
    assert "我不知道" not in second_prompt

    calls = (
        await db.execute(select(ToolCall).where(ToolCall.session_id == session.id))
    ).scalars().all()
    assert len(calls) == 1
    assert calls[0].tool_name == "search"
    assert calls[0].status == "ok"
    assert calls[0].result_meta["citations"]

    messages = (
        await db.execute(
            select(Message)
            .where(Message.session_id == session.id)
            .order_by(Message.created_at.asc())
        )
    ).scalars().all()
    final = [item for item in messages if item.meta.get("kind") == "final"][-1]
    assert final.meta["forced_search"] is True
    assert "42" in final.content
    assert not any(item.role == "assistant" and item.content == "我不知道。" for item in messages)

    event = (await _events(db, session.id, "search_forced"))[-1]
    assert event.payload["status"] == "ok"
    assert event.payload["result_count"] == 1
    assert event.payload["reachable_count"] == 1
    assert event.payload["query_chars"] == len(user_input)
    assert event.payload["latency_ms"] >= 0

    turn_event = (await _events(db, session.id, "turn"))[-1]
    assert turn_event.payload.get("first_token_ms") is not None


async def test_forced_search_empty_results_returns_fixed_text(
    settings: Settings, db: Any
) -> None:
    session = await create_session(db, settings, "空结果")
    provider = ScriptedProvider(scripts=["Thought: 不知道\nFinal Answer: 我不知道。"])
    mcp = FakeMCP(responses={"search": _search_payload(empty=True)})
    loop, _ = build_loop(settings, provider, mcp)
    collector = Collector()

    result = await loop.run(session.id, "查一下火星人口", collector)

    assert result.finish_reason == "final_answer"
    assert result.final_answer == FORCED_SEARCH_EMPTY_TEXT
    assert _deltas(collector) == FORCED_SEARCH_EMPTY_TEXT
    assert "我不知道" not in _deltas(collector)
    assert len(provider.calls) == 1
    event = (await _events(db, session.id, "search_forced"))[-1]
    assert event.payload["status"] == "empty"
    final = (
        await db.execute(
            select(Message)
            .where(Message.session_id == session.id, Message.role == "assistant")
            .order_by(Message.created_at.desc())
            .limit(1)
        )
    ).scalar_one()
    assert final.content == FORCED_SEARCH_EMPTY_TEXT
    assert final.meta["forced_search"] is True
    assert final.meta["replace_reason"] == "search_empty"


async def test_forced_search_error_returns_fixed_text(settings: Settings, db: Any) -> None:
    session = await create_session(db, settings, "检索报错")
    provider = ScriptedProvider(scripts=["Thought: 不知道\nFinal Answer: 我不知道。"])
    mcp = FakeMCP(fail_tools={"search"})
    loop, _ = build_loop(settings, provider, mcp)
    collector = Collector()

    result = await loop.run(session.id, "查一下火星人口", collector)

    assert result.final_answer == FORCED_SEARCH_EMPTY_TEXT
    assert "我不知道" not in _deltas(collector)
    event = (await _events(db, session.id, "search_forced"))[-1]
    assert event.payload["status"] == "error"
    assert event.payload["error"]
    calls = (
        await db.execute(select(ToolCall).where(ToolCall.session_id == session.id))
    ).scalars().all()
    assert len(calls) == 1
    assert calls[0].status == "error"


async def test_forced_search_skipped_when_mcp_offline(settings: Settings, db: Any) -> None:
    session = await create_session(db, settings, "工具离线")
    provider = ScriptedProvider(scripts=["Thought: 不知道\nFinal Answer: 我不知道。"])
    mcp = FakeMCP(started=False)
    loop, _ = build_loop(settings, provider, mcp)
    collector = Collector()

    result = await loop.run(session.id, "查一下火星人口", collector)

    assert result.final_answer == FORCED_SEARCH_EMPTY_TEXT
    assert "tool_start" not in collector.names()
    skipped = (await _events(db, session.id, "search_forced_skipped"))[-1]
    assert skipped.payload["reason"] == "mcp_offline"
    final = (
        await db.execute(
            select(Message)
            .where(Message.session_id == session.id, Message.role == "assistant")
            .order_by(Message.created_at.desc())
            .limit(1)
        )
    ).scalar_one()
    assert final.meta["replace_reason"] == "search_skipped:mcp_offline"


async def test_existing_successful_search_does_not_trigger_again(
    settings: Settings, db: Any
) -> None:
    session = await create_session(db, settings, "已有搜索")
    provider = ScriptedProvider(
        scripts=[
            'Thought: 先搜索\nAction: search\nAction Input: {"query": "火星人口"}',
            "Thought: 仍然不知道\nFinal Answer: 我不知道。",
        ]
    )
    mcp = FakeMCP(responses={"search": _search_payload()})
    loop, _ = build_loop(settings, provider, mcp)
    collector = Collector()

    result = await loop.run(session.id, "查一下火星人口", collector)

    assert result.finish_reason == "final_answer"
    assert result.final_answer == "我不知道。"
    assert len(mcp.calls) == 1
    assert not await _events(db, session.id, "search_forced")


async def test_successful_empty_search_also_suppresses_forced_search(
    settings: Settings, db: Any
) -> None:
    session = await create_session(db, settings, "已有空搜索")
    provider = ScriptedProvider(
        scripts=[
            'Thought: 先搜索\nAction: search\nAction Input: {"query": "火星人口"}',
            "Thought: 仍然不知道\nFinal Answer: 我不知道。",
        ]
    )
    mcp = FakeMCP(responses={"search": _search_payload(empty=True)})
    loop, _ = build_loop(settings, provider, mcp)
    collector = Collector()

    result = await loop.run(session.id, "查一下火星人口", collector)

    assert result.finish_reason == "final_answer"
    assert result.final_answer == "我不知道。"
    assert len(mcp.calls) == 1
    assert not await _events(db, session.id, "search_forced")


async def test_forced_search_skipped_on_task_turn_limit(settings: Settings, db: Any) -> None:
    session = await create_session(db, settings, "30轮上限")
    session.task_turn_count = 30
    await db.commit()
    provider = ScriptedProvider(scripts=["Thought: 不知道\nFinal Answer: 我不知道。"])
    mcp = FakeMCP(responses={"search": _search_payload()})
    loop, _ = build_loop(settings, provider, mcp)
    collector = Collector()

    result = await loop.run(session.id, "查一下火星人口", collector)

    assert result.finish_reason == "final_answer"
    assert result.final_answer == FORCED_SEARCH_EMPTY_TEXT
    assert not mcp.calls
    skipped = (await _events(db, session.id, "search_forced_skipped"))[-1]
    assert skipped.payload["reason"] == "max_task_turns"
    await db.refresh(session)
    assert session.status == "active"
    assert session.task_turn_count == 30
    assert collector.first("error") is None
    final = (
        await db.execute(
            select(Message)
            .where(Message.session_id == session.id, Message.role == "assistant")
            .order_by(Message.created_at.desc())
            .limit(1)
        )
    ).scalar_one()
    assert final.content == FORCED_SEARCH_EMPTY_TEXT
    assert final.meta["kind"] == "final"
    assert final.meta["forced_search"] is True
    assert final.meta["replace_reason"] == "search_skipped:max_task_turns"


async def test_forced_search_replaces_second_refusal(settings: Settings, db: Any) -> None:
    session = await create_session(db, settings, "重答仍拒绝")
    provider = ScriptedProvider(
        scripts=[
            "Thought: 不知道\nFinal Answer: 我不知道。",
            "Thought: 还是不知道\nFinal Answer: 很抱歉，没有相关信息。",
        ]
    )
    mcp = FakeMCP(responses={"search": _search_payload()})
    loop, _ = build_loop(settings, provider, mcp)
    collector = Collector()

    result = await loop.run(session.id, "查一下火星人口", collector)

    assert result.finish_reason == "final_answer"
    assert result.final_answer == FORCED_SEARCH_EMPTY_TEXT
    assert len(provider.calls) == 2
    assert len(mcp.calls) == 1
    assert "没有相关信息" not in _deltas(collector)
    replaced = await _events(db, session.id, "forced_answer_replaced")
    assert replaced
    assert replaced[-1].payload["reason"] == "refusal_after_search"
    final = (
        await db.execute(
            select(Message)
            .where(Message.session_id == session.id, Message.role == "assistant")
            .order_by(Message.created_at.desc())
            .limit(1)
        )
    ).scalar_one()
    assert final.meta["replace_reason"] == "refusal_after_search"


async def test_native_mode_forced_search_round_trip(settings: Settings, db: Any) -> None:
    session = await create_session(db, settings, "native强制检索")
    session.mode = "native"
    await db.commit()
    provider = ScriptedProvider(
        scripts=[
            {"text": "I don't know."},
            {"text": "Based on the search, the answer is 42."},
        ]
    )
    mcp = FakeMCP(responses={"search": _search_payload()})
    loop, _ = build_loop(settings, provider, mcp)
    collector = Collector()

    result = await loop.run(session.id, "Mars population 2026", collector)

    assert result.finish_reason == "final_answer"
    assert "42" in result.final_answer
    assert "I don't know" not in _deltas(collector)
    assert collector.names().count("tool_start") == 1
    assert collector.first("tool_result")["citations"]
    second_prompt = provider.calls[1]
    roles = [message.get("role") for message in second_prompt[-2:]]
    assert roles == ["assistant", "tool"]
    assert second_prompt[-2]["tool_calls"][0]["function"]["name"] == "search"
    assert "火星人口白皮书" in str(second_prompt[-1]["content"])


async def test_native_refusal_preamble_is_not_exposed(settings: Settings, db: Any) -> None:
    session = await create_session(db, settings, "native拒绝正文")
    session.mode = "native"
    await db.commit()
    provider = ScriptedProvider(
        scripts=[
            {
                "text": "I don't know.",
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": '{"query": "火星人口"}',
                        },
                    }
                ],
            },
            {"text": "Based on the search, the answer is 42."},
        ]
    )
    mcp = FakeMCP(responses={"search": _search_payload()})
    loop, _ = build_loop(settings, provider, mcp)
    collector = Collector()

    result = await loop.run(session.id, "Mars population", collector)

    assert result.finish_reason == "final_answer"
    assert "I don't know" not in _deltas(collector)
    assert "42" in result.final_answer
    assert len(mcp.calls) == 1
    second_prompt = provider.calls[1]
    action_message = next(
        message
        for message in reversed(second_prompt)
        if message.get("role") == "assistant"
    )
    assert action_message["content"] == ""
    assert action_message["tool_calls"][0]["function"]["name"] == "search"


async def test_forced_search_intercepts_refusal_when_switch_off(
    settings: Settings, db: Any
) -> None:
    session = await create_session(db, settings, "开关关闭")
    disabled = settings.model_copy(update={"forced_search_enabled": False})
    provider = ScriptedProvider(scripts=["Thought: 不知道\nFinal Answer: 我不知道。"])
    mcp = FakeMCP(responses={"search": _search_payload()})
    loop, _ = build_loop(disabled, provider, mcp)
    collector = Collector()

    result = await loop.run(session.id, "查一下火星人口", collector)

    assert result.final_answer == FORCED_SEARCH_EMPTY_TEXT
    assert not mcp.calls
    assert "我不知道" not in _deltas(collector)
    skipped = (await _events(db, session.id, "search_forced_skipped"))[-1]
    assert skipped.payload["reason"] == "disabled"
    assert skipped.payload["trigger"] == "uncertainty"
    assert skipped.payload["query_chars"] == len("查一下火星人口")


async def test_forced_search_skips_disabled_tool(settings: Settings, db: Any) -> None:
    session = await create_session(db, settings, "工具禁用")
    session.blocked_tools = ["search"]
    await db.commit()
    provider = ScriptedProvider(scripts=["Thought: 不知道\nFinal Answer: 我不知道。"])
    mcp = FakeMCP(responses={"search": _search_payload()})
    loop, _ = build_loop(settings, provider, mcp)
    collector = Collector()

    result = await loop.run(session.id, "查一下火星人口", collector)

    assert result.final_answer == FORCED_SEARCH_EMPTY_TEXT
    assert not mcp.calls
    skipped = (await _events(db, session.id, "search_forced_skipped"))[-1]
    assert skipped.payload["reason"] == "tool_disabled"


async def test_forced_search_skips_missing_registry_entry(
    settings: Settings, db: Any
) -> None:
    session = await create_session(db, settings, "注册表缺失")
    provider = ScriptedProvider(scripts=["Thought: 不知道\nFinal Answer: 我不知道。"])
    mcp = FakeMCP(tools=[registry()["calculator"]])
    loop, _ = build_loop(settings, provider, mcp)
    collector = Collector()

    result = await loop.run(session.id, "查一下火星人口", collector)

    assert result.final_answer == FORCED_SEARCH_EMPTY_TEXT
    assert not mcp.calls
    skipped = (await _events(db, session.id, "search_forced_skipped"))[-1]
    assert skipped.payload["reason"] == "tool_missing"


async def test_forced_search_skips_when_tool_iterations_exhausted(
    settings: Settings, db: Any
) -> None:
    session = await create_session(db, settings, "工具迭代上限")
    limited = settings.model_copy(update={"max_tool_iterations": 0})
    provider = ScriptedProvider(scripts=["Thought: 不知道\nFinal Answer: 我不知道。"])
    mcp = FakeMCP(responses={"search": _search_payload()})
    loop, _ = build_loop(limited, provider, mcp)
    collector = Collector()

    result = await loop.run(session.id, "查一下火星人口", collector)

    assert result.final_answer == FORCED_SEARCH_EMPTY_TEXT
    assert not mcp.calls
    skipped = (await _events(db, session.id, "search_forced_skipped"))[-1]
    assert skipped.payload["reason"] == "max_tool_iterations"


async def test_forced_search_intercepts_when_max_per_turn_is_zero(
    settings: Settings, db: Any
) -> None:
    session = await create_session(db, settings, "每轮零次")
    zero = settings.model_copy(update={"forced_search_max_per_turn": 0})
    provider = ScriptedProvider(scripts=["Thought: 不知道\nFinal Answer: 我不知道。"])
    mcp = FakeMCP(responses={"search": _search_payload()})
    loop, _ = build_loop(zero, provider, mcp)
    collector = Collector()

    result = await loop.run(session.id, "查一下火星人口", collector)

    assert result.final_answer == FORCED_SEARCH_EMPTY_TEXT
    assert not mcp.calls
    skipped = (await _events(db, session.id, "search_forced_skipped"))[-1]
    assert skipped.payload["reason"] == "max_per_turn_reached"


async def test_forced_search_budget_exhausted_still_wraps_up(
    settings: Settings, db: Any
) -> None:
    session = await create_session(db, settings, "预算不足重答")
    limited = settings.model_copy(update={"max_model_calls_per_turn": 1})
    provider = ScriptedProvider(scripts=["Thought: 不知道\nFinal Answer: 我不知道。"])
    mcp = FakeMCP(responses={"search": _search_payload()})
    loop, _ = build_loop(limited, provider, mcp)
    collector = Collector()

    result = await loop.run(session.id, "查一下火星人口", collector)

    assert result.finish_reason == "max_model_calls"
    assert "我不知道" not in _deltas(collector)
    assert len(mcp.calls) == 1
    final = (
        await db.execute(
            select(Message)
            .where(Message.session_id == session.id, Message.role == "assistant")
            .order_by(Message.created_at.desc())
            .limit(1)
        )
    ).scalar_one()
    assert final.meta["kind"] == "wrap_up"
    assert final.meta["forced_search"] is True
    assert "search" in final.content


async def test_unreachable_citations_still_count_as_results(
    settings: Settings, db: Any
) -> None:
    session = await create_session(db, settings, "引用不可达")
    provider = ScriptedProvider(
        scripts=[
            "Thought: 不知道\nFinal Answer: 我不知道。",
            "Thought: 已检索\nFinal Answer: 根据检索，答案是 42。",
        ]
    )
    mcp = FakeMCP(
        responses={"search": _search_payload(url="http://127.0.0.1:9/private")}
    )
    loop, _ = build_loop(settings, provider, mcp)
    collector = Collector()

    result = await loop.run(session.id, "查一下火星人口", collector)

    assert result.finish_reason == "final_answer"
    assert "42" in result.final_answer
    event = (await _events(db, session.id, "search_forced"))[-1]
    assert event.payload["status"] == "ok"
    assert event.payload["reachable_count"] == 0


async def test_buffer_limit_releases_answer_without_forcing_search(
    settings: Settings, db: Any
) -> None:
    session = await create_session(db, settings, "缓冲超限")
    tiny = settings.model_copy(update={"final_answer_buffer_max": 10})
    answer = "我不知道。" + "x" * 20
    provider = ScriptedProvider(
        scripts=[f"Thought: 不知道\nFinal Answer: {answer}"]
    )
    mcp = FakeMCP(responses={"search": _search_payload()})
    loop, _ = build_loop(tiny, provider, mcp)
    collector = Collector()

    result = await loop.run(session.id, "查一下火星人口", collector)

    assert result.finish_reason == "final_answer"
    assert result.final_answer == answer
    assert "我不知道" in _deltas(collector)
    assert not mcp.calls
    assert not await _events(db, session.id, "search_forced")


async def test_forced_search_query_is_truncated_to_500(settings: Settings, db: Any) -> None:
    session = await create_session(db, settings, "检索词截断")
    roomy = settings.model_copy(update={"context_compress_threshold": 100_000})
    provider = ScriptedProvider(scripts=["Thought: 不知道\nFinal Answer: 我不知道。"])
    mcp = FakeMCP(responses={"search": _search_payload(empty=True)})
    loop, _ = build_loop(roomy, provider, mcp)
    collector = Collector()
    user_input = "火星人口" + "x" * 600

    await loop.run(session.id, user_input, collector)

    assert len(mcp.calls) == 1
    query = mcp.calls[0][1]["query"]
    assert query == user_input.strip()[:500]
    assert len(query) == 500


async def test_admin_metrics_endpoint_exposes_first_token_fields(
    settings: Settings,
) -> None:
    provider = ScriptedProvider(
        scripts=["Thought: 回答\nFinal Answer: 结果是 2。"]
    )
    gateway = ModelGateway(settings, providers=[provider], tracer=get_tracer(settings))
    app = create_app(settings, gateway=gateway, mcp=FakeMCP(), auto_start_mcp=True)

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            session_id = (await client.post("/api/sessions", json={})).json()["id"]
            response = await client.post(
                f"/api/sessions/{session_id}/messages", json={"content": "1+1"}
            )
            assert response.status_code == 200

            now = datetime.now(UTC)
            metrics = await client.get(
                "/api/admin/metrics",
                params={
                    "start": (now - timedelta(minutes=10)).isoformat(),
                    "end": (now + timedelta(minutes=2)).isoformat(),
                },
            )
            assert metrics.status_code == 200
            payload = metrics.json()
            assert "p95_first_token_ms" in payload["overview"]
            assert payload["overview"]["p95_first_token_ms"] is not None
            assert all("p95_first_token_ms" in point for point in payload["trend"])
            deepseek = next(
                item for item in payload["models"] if item["layer"] == "deepseek"
            )
            assert deepseek["first_token_samples"] >= 1
            assert deepseek["first_token_p95_ms"] is not None


async def test_forced_search_sse_order_and_fixed_text(settings: Settings) -> None:
    provider = ScriptedProvider(scripts=["Thought: 不知道\nFinal Answer: 我不知道。"])
    mcp = FakeMCP(responses={"search": _search_payload(empty=True)})
    gateway = ModelGateway(settings, providers=[provider], tracer=get_tracer(settings))
    app = create_app(settings, gateway=gateway, mcp=mcp, auto_start_mcp=True)

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            session_id = (await client.post("/api/sessions", json={})).json()["id"]
            response = await client.post(
                f"/api/sessions/{session_id}/messages",
                json={"content": "查一下火星人口"},
            )
            assert response.status_code == 200
            body = response.text
            assert "event: tool_result" in body
            assert FORCED_SEARCH_EMPTY_TEXT in body
            assert "我不知道" not in body
            assert body.index("event: tool_result") < body.index(
                FORCED_SEARCH_EMPTY_TEXT
            )


async def test_forced_search_error_over_http_returns_fixed_text(settings: Settings) -> None:
    provider = ScriptedProvider(scripts=["Thought: 不知道\nFinal Answer: 我不知道。"])
    mcp = FakeMCP(fail_tools={"search"})
    gateway = ModelGateway(settings, providers=[provider], tracer=get_tracer(settings))
    app = create_app(settings, gateway=gateway, mcp=mcp, auto_start_mcp=True)

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            session_id = (await client.post("/api/sessions", json={})).json()["id"]
            response = await client.post(
                f"/api/sessions/{session_id}/messages",
                json={"content": "查一下火星人口"},
            )
            assert response.status_code == 200
            assert FORCED_SEARCH_EMPTY_TEXT in response.text
            assert "我不知道" not in response.text


async def test_forced_search_offline_over_http_returns_fixed_text(
    settings: Settings,
) -> None:
    provider = ScriptedProvider(scripts=["Thought: 不知道\nFinal Answer: 我不知道。"])
    mcp = FakeMCP(started=False)
    gateway = ModelGateway(settings, providers=[provider], tracer=get_tracer(settings))
    app = create_app(settings, gateway=gateway, mcp=mcp, auto_start_mcp=False)

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            session_id = (await client.post("/api/sessions", json={})).json()["id"]
            response = await client.post(
                f"/api/sessions/{session_id}/messages",
                json={"content": "查一下火星人口"},
            )
            assert response.status_code == 200
            assert FORCED_SEARCH_EMPTY_TEXT in response.text


async def test_forced_search_disabled_tool_over_http_returns_fixed_text(
    settings: Settings, db: Any
) -> None:
    provider = ScriptedProvider(scripts=["Thought: 不知道\nFinal Answer: 我不知道。"])
    mcp = FakeMCP(responses={"search": _search_payload()})
    gateway = ModelGateway(settings, providers=[provider], tracer=get_tracer(settings))
    app = create_app(settings, gateway=gateway, mcp=mcp, auto_start_mcp=True)

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            session_id = (await client.post("/api/sessions", json={})).json()["id"]
            session = await db.get(Session, uuid.UUID(session_id))
            assert session is not None
            session.blocked_tools = ["search"]
            await db.commit()

            response = await client.post(
                f"/api/sessions/{session_id}/messages",
                json={"content": "查一下火星人口"},
            )
            assert response.status_code == 200
            assert FORCED_SEARCH_EMPTY_TEXT in response.text


async def test_admin_metrics_empty_samples_return_null(settings: Settings) -> None:
    gateway = ModelGateway(settings, providers=[], tracer=get_tracer(settings))
    app = create_app(settings, gateway=gateway, mcp=FakeMCP(), auto_start_mcp=True)

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            now = datetime.now(UTC)
            metrics = await client.get(
                "/api/admin/metrics",
                params={
                    "start": (now - timedelta(hours=1)).isoformat(),
                    "end": (now + timedelta(minutes=2)).isoformat(),
                },
            )
            assert metrics.status_code == 200
            payload = metrics.json()
            assert payload["overview"]["p95_first_token_ms"] is None
            assert payload["models"] == []
            assert all(
                point["p95_first_token_ms"] is None for point in payload["trend"]
            )


async def test_admin_metrics_first_token_p95(settings: Settings, db: Any) -> None:
    session = await create_session(db, settings, "首Token指标")
    now = datetime.now(UTC)
    for index, value in enumerate([100, 200, 300]):
        db.add(
            Event(
                trace_id=uuid.uuid4().hex,
                span_id=uuid.uuid4().hex[:16],
                parent_span_id=None,
                session_id=str(session.id),
                turn_no=index + 1,
                kind="turn",
                payload={"first_token_ms": value},
                duration_ms=1000,
                ts=now - timedelta(minutes=10 + index),
            )
        )
    db.add(
        Event(
            trace_id=uuid.uuid4().hex,
            span_id=uuid.uuid4().hex[:16],
            parent_span_id=None,
            session_id=str(session.id),
            turn_no=9,
            kind="turn",
            payload={},
            duration_ms=1000,
            ts=now - timedelta(minutes=5),
        )
    )
    for value in (50, 150):
        db.add(
            Event(
                trace_id=uuid.uuid4().hex,
                span_id=uuid.uuid4().hex[:16],
                parent_span_id=None,
                session_id=str(session.id),
                turn_no=1,
                kind="model_success",
                payload={"layer": "deepseek", "first_token_ms": value},
                duration_ms=0,
                ts=now - timedelta(minutes=4),
            )
        )
    db.add(
        Event(
            trace_id=uuid.uuid4().hex,
            span_id=uuid.uuid4().hex[:16],
            parent_span_id=None,
            session_id=str(session.id),
            turn_no=1,
            kind="model_success",
            payload={"layer": "deepseek"},
            duration_ms=0,
            ts=now - timedelta(minutes=3),
        )
    )
    await db.commit()

    metrics = await compute_metrics(
        db, settings, now - timedelta(hours=1), now + timedelta(minutes=1)
    )

    assert metrics["overview"]["p95_first_token_ms"] == 300.0
    assert any(
        point["p95_first_token_ms"] == 300.0 for point in metrics["trend"]
    )
    deepseek = next(item for item in metrics["models"] if item["layer"] == "deepseek")
    assert deepseek["first_token_samples"] == 2
    assert deepseek["first_token_p95_ms"] == 150.0

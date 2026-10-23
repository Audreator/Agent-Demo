from __future__ import annotations

import asyncio
from typing import Any

from sqlalchemy import select

from app.agent.loop import AgentLoop
from app.config import Settings
from app.models.db_models import Event, Message, StateCard, Summary, ToolCall
from app.models.gateway import ModelGateway
from app.observability.tracing import get_tracer
from app.sessions.repository import SessionRepository
from tests.helpers import Collector, FakeMCP, ScriptedProvider, registry


async def _no_sleep(_: float) -> None:
    await asyncio.sleep(0)


def build_loop(
    settings: Settings,
    provider: Any,
    mcp: FakeMCP | None = None,
) -> tuple[AgentLoop, FakeMCP]:
    fake_mcp = mcp or FakeMCP()
    gateway = ModelGateway(
        settings,
        providers=[provider],
        tracer=get_tracer(settings),
        sleep=_no_sleep,
        jitter=lambda: 0.0,
    )
    loop = AgentLoop(settings, gateway, fake_mcp, user_id=settings.dev_user_id)  # type: ignore[arg-type]
    return loop, fake_mcp


async def create_session(db: Any, settings: Settings, title: str = "测试会话"):
    repo = SessionRepository(db, settings.dev_user_id)
    session = await repo.create(title=title)
    await db.commit()
    return session


async def test_two_tool_calls_close_loop(settings: Settings, db: Any) -> None:
    session = await create_session(db, settings, "新会话")
    provider = ScriptedProvider(
        scripts=[
            "Thought: 先计算\nAction: calculator\nAction Input: {\"expression\": \"1+2\"}",
            "Thought: 再建待办\nAction: todo\nAction Input: "
            '{"action": "create", "content": "复盘", "client_op_id": "op-1"}',
            "Thought: 完成了\nFinal Answer: 结果是 3，待办已创建。",
        ]
    )
    mcp = FakeMCP(
        responses={
            "calculator": {"expression": "1+2", "result": "3", "formatted": "3"},
            "todo": {
                "todo": {
                    "id": "11111111-1111-1111-1111-111111111111",
                    "content": "复盘",
                    "status": "open",
                    "created_at": "2026-10-21T00:00:00+00:00",
                    "updated_at": "2026-10-21T00:00:00+00:00",
                }
            },
        }
    )
    loop, _ = build_loop(settings, provider, mcp)
    collector = Collector()
    result = await loop.run(session.id, "帮我算 1+2 并建一个待办", collector)

    assert result.finish_reason == "final_answer"
    assert "3" in result.final_answer
    assert collector.names().count("tool_start") == 2
    assert collector.names().count("tool_result") == 2
    assert collector.names().count("done") == 1
    assert collector.first("done")["finish_reason"] == "final_answer"

    messages = (
        await db.execute(
            select(Message)
            .where(Message.session_id == session.id)
            .order_by(Message.turn_no, Message.created_at)
        )
    ).scalars().all()
    assert [message.role for message in messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
    ]
    assert messages[-1].content.startswith("结果是 3")

    calls = (
        await db.execute(select(ToolCall).where(ToolCall.session_id == session.id))
    ).scalars().all()
    assert len(calls) == 2
    assert all(call.status == "ok" for call in calls)
    assert calls[0].tool_name == "calculator"
    assert calls[1].tool_name == "todo"
    assert calls[1].idempotency_key == "op-1"

    await db.refresh(session)
    refreshed = session
    assert refreshed is not None
    assert refreshed.task_turn_count == 1
    assert refreshed.title.startswith("帮我算")
    assert refreshed.mode == "react"

    card = (
        await db.execute(select(StateCard).where(StateCard.session_id == session.id))
    ).scalar_one()
    await db.refresh(card)
    assert card.goal
    assert len(card.confirmed_facts) >= 2
    assert any("复盘" in str(item) for item in card.todos)
    assert card.version >= 1

    events = (
        await db.execute(select(Event).where(Event.trace_id == result.trace_id))
    ).scalars().all()
    kinds = {event.kind for event in events}
    assert {"turn", "context_build", "model_call", "tool_call"} <= kinds
    assert all(event.trace_id == result.trace_id for event in events)
    turn_event = next(event for event in events if event.kind == "turn")
    request_event = next(event for event in events if event.kind == "request")
    assert request_event.parent_span_id is None
    assert turn_event.parent_span_id == request_event.span_id
    model_events = [event for event in events if event.kind == "model_call"]
    tool_events = [event for event in events if event.kind == "tool_call"]
    assert model_events and all(
        event.parent_span_id == turn_event.span_id for event in model_events
    )
    assert tool_events and all(
        event.parent_span_id == turn_event.span_id for event in tool_events
    )


async def test_token_budget_compression_and_follow_up(settings: Settings, db: Any) -> None:
    session = await create_session(db, settings, "压缩")
    repo = SessionRepository(db, settings.dev_user_id)
    for turn in range(1, 7):
        user_text = f"第{turn}轮请求 " + ("u" * 135)
        if turn == 1:
            user_text = "OLD-FACT: 我的偏好是深色模式 " + ("u" * 120)
        await repo.add_message(
            session_id=session.id,
            turn_no=turn,
            role="user",
            content=user_text,
        )
        await repo.add_message(
            session_id=session.id,
            turn_no=turn,
            role="assistant",
            content=f"第{turn}轮答复 " + ("a" * 135),
        )
    await repo.add_message(
        session_id=session.id,
        turn_no=6,
        role="tool",
        content="工具原始结果 " + ("t" * 1480),
        meta={
            "type": "tool_result",
            "tool_name": "read_docs",
            "status": "ok",
            "summary": "文档摘要",
            "result_ref": "artifact://big",
        },
    )
    await db.commit()

    provider = ScriptedProvider(
        scripts=[
            "Thought: 我知道偏好\nFinal Answer: 你的偏好是深色模式，已记住。",
            "Thought: 继续\nFinal Answer: 追问回答：深色模式。",
        ]
    )
    fake_mcp = FakeMCP(tools=[registry()["calculator"]])
    loop, _ = build_loop(settings, provider, fake_mcp)

    collector = Collector()
    first = await loop.run(session.id, "请根据历史回答我的偏好", collector)
    assert first.finish_reason == "final_answer"
    context_event = collector.first("context_update")
    assert context_event is not None
    assert context_event["compressed"] is True
    assert context_event["tokens_before"] > 2000
    assert context_event["tokens_after"] <= context_event["threshold"]

    summary_rows = (
        await db.execute(select(Summary).where(Summary.session_id == session.id))
    ).scalars().all()
    assert summary_rows
    assert "深色模式" in "\n".join(row.content for row in summary_rows)

    first_prompt = provider.calls[0]
    rendered = "\n".join(str(message.get("content")) for message in first_prompt)
    assert "历史摘要" in rendered
    assert "OLD-FACT" in rendered
    assert "第6轮请求" in rendered

    collector2 = Collector()
    second = await loop.run(session.id, "我的偏好是什么？继续", collector2)
    assert second.finish_reason == "final_answer"
    second_prompt = provider.calls[-1]
    second_rendered = "\n".join(str(message.get("content")) for message in second_prompt)
    assert "历史摘要" in second_rendered
    assert "深色模式" in second_rendered
    assert "追问回答：深色模式" in second.final_answer


async def test_tool_consecutive_failures_disable_tool(settings: Settings, db: Any) -> None:
    session = await create_session(db, settings, "工具失败")
    provider = ScriptedProvider(
        scripts=[
            'Thought: 查天气\nAction: weather\nAction Input: {"location": "杭州"}',
            'Thought: 再试\nAction: weather\nAction Input: {"location": "杭州"}',
            'Thought: 再试\nAction: weather\nAction Input: {"location": "杭州"}',
            'Thought: 再试\nAction: weather\nAction Input: {"location": "杭州"}',
            "Thought: 放弃工具\nFinal Answer: 天气服务暂时不可用。",
        ]
    )
    mcp = FakeMCP(fail_tools={"weather"})
    loop, _ = build_loop(settings, provider, mcp)
    collector = Collector()
    result = await loop.run(session.id, "查一下杭州天气", collector)

    assert result.finish_reason == "final_answer"
    assert len(mcp.calls) == 3
    await db.refresh(session)
    refreshed = session
    assert refreshed is not None
    assert "weather" in refreshed.blocked_tools
    assert refreshed.tool_failure_streaks["weather"] == 3

    calls = (
        await db.execute(select(ToolCall).where(ToolCall.session_id == session.id))
    ).scalars().all()
    assert len(calls) == 4
    assert sum(1 for call in calls if call.status == "error") == 3
    blocked_calls = [call for call in calls if call.status == "blocked"]
    assert len(blocked_calls) == 1
    assert blocked_calls[0].tool_name == "weather"
    statuses = [data.get("status") for name, data in collector.events if name == "tool_result"]
    assert statuses.count("error") == 3
    assert statuses.count("blocked") == 1
    assert refreshed.task_turn_count == 1

    card = (
        await db.execute(select(StateCard).where(StateCard.session_id == session.id))
    ).scalar_one()
    await db.refresh(card)
    assert card.open_questions


async def test_native_fallback_switch_and_persist(settings: Settings, db: Any) -> None:
    session = await create_session(db, settings, "native")
    provider = ScriptedProvider(
        scripts=[
            "这不是合法格式，只有一段普通文本。",
            "仍然没有 Action 或 Final Answer。",
            "第二轮依然格式错误。",
            "依旧没有任何字段。",
            {"reasoning": "原生模式思考", "text": "Native 最终答案：完成了。"},
        ]
    )
    loop, _ = build_loop(settings, provider)

    first = Collector()
    await loop.run(session.id, "第一轮", first)
    await db.refresh(session)
    refreshed = session
    assert refreshed is not None
    assert refreshed.mode == "react"
    assert refreshed.parse_failures == 1

    second = Collector()
    result = await loop.run(session.id, "第二轮", second)
    assert result.finish_reason == "parse_error"
    await db.refresh(session)
    refreshed = session
    assert refreshed is not None
    assert refreshed.mode == "native"
    assert refreshed.parse_failures == 0
    assert any(
        name == "context_update" and data.get("mode") == "native"
        for name, data in second.events
    )

    third = Collector()
    result = await loop.run(session.id, "第三轮", third)
    assert result.finish_reason == "final_answer"
    assert "Native 最终答案" in result.final_answer
    await db.refresh(session)
    refreshed = session
    assert refreshed is not None
    assert refreshed.mode == "native"

    final_message = (
        await db.execute(
            select(Message)
            .where(Message.session_id == session.id, Message.role == "assistant")
            .order_by(Message.created_at.desc())
            .limit(1)
        )
    ).scalar_one()
    assert final_message.meta.get("thought") == "原生模式思考"


async def test_budget_exceeded_wraps_with_user_notice(settings: Settings, db: Any) -> None:
    session = await create_session(db, settings, "预算")
    provider = ScriptedProvider(scripts=["Thought: 不应调用\nFinal Answer: no"])
    loop, _ = build_loop(settings, provider)
    collector = Collector()
    result = await loop.run(session.id, "x" * 6000, collector)

    assert result.finish_reason == "budget_exceeded"
    assert not provider.calls
    assert collector.first("error")["code"] == "BUDGET_EXCEEDED"
    assert collector.first("done")["finish_reason"] == "budget_exceeded"
    assert any(name == "message_delta" for name, _ in collector.events)

    messages = (
        await db.execute(select(Message).where(Message.session_id == session.id))
    ).scalars().all()
    assert messages[-1].role == "assistant"
    assert "预算" in messages[-1].content


async def test_pure_chat_is_allowed_after_task_limit(settings: Settings, db: Any) -> None:
    session = await create_session(db, settings, "上限-纯聊天")
    session.task_turn_count = 30
    await db.commit()

    provider = ScriptedProvider(
        scripts=["Thought: 纯聊天\nFinal Answer: 这是不带工具的普通回答。"]
    )
    loop, _ = build_loop(settings, provider)
    collector = Collector()
    result = await loop.run(session.id, "我们随便聊聊", collector)

    assert result.finish_reason == "final_answer"
    assert "普通回答" in result.final_answer
    assert len(provider.calls) == 1
    assert collector.first("error") is None
    await db.refresh(session)
    assert session.task_turn_count == 30


async def test_31st_task_turn_rejected_and_wrapped(settings: Settings, db: Any) -> None:
    session = await create_session(db, settings, "上限-任务")
    session.task_turn_count = 30
    await db.commit()

    provider = ScriptedProvider(
        scripts=[
            "Thought: 再算一次\nAction: calculator\nAction Input: {\"expression\": \"1+1\"}",
            "Thought: 收束\nFinal Answer: 已到任务轮次上限，停止继续调用工具。",
        ]
    )
    mcp = FakeMCP(
        responses={"calculator": {"expression": "1+1", "result": "2", "formatted": "2"}}
    )
    loop, _ = build_loop(settings, provider, mcp)
    collector = Collector()
    result = await loop.run(session.id, "继续算 1+1", collector)

    assert result.finish_reason == "max_task_turns"
    assert not mcp.calls
    assert collector.first("error")["code"] == "TASK_TURN_LIMIT"
    assert collector.first("done")["finish_reason"] == "max_task_turns"
    await db.refresh(session)
    assert session.task_turn_count == 30
    assert session.status == "wrapped"

    messages = (
        await db.execute(select(Message).where(Message.session_id == session.id))
    ).scalars().all()
    assert len(messages) == 2
    assert messages[-1].role == "assistant"
    assert messages[-1].content

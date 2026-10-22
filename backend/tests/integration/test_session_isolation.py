from __future__ import annotations

import asyncio
from typing import Any

from app.config import Settings
from app.sessions.repository import SessionRepository
from tests.helpers import Collector, EchoProvider
from tests.integration.test_loop import build_loop, create_session


async def test_two_sessions_do_not_cross_talk(settings: Settings, db: Any) -> None:
    session_a = await create_session(db, settings, "A")
    session_b = await create_session(db, settings, "B")
    provider = EchoProvider()
    loop, _ = build_loop(settings, provider)

    await loop.run(session_a.id, "A-一号 秘密内容", Collector())
    await loop.run(session_b.id, "B-一号 另一件事", Collector())
    await loop.run(session_a.id, "A-二号 继续", Collector())
    await loop.run(session_b.id, "B-二号 继续", Collector())

    repo = SessionRepository(db, settings.dev_user_id)
    messages_a = await repo.list_messages(session_a.id)
    messages_b = await repo.list_messages(session_b.id)

    text_a = "\n".join(message.content for message in messages_a)
    text_b = "\n".join(message.content for message in messages_b)
    assert "B-一号" not in text_a
    assert "A-一号" not in text_b
    assert "A-一号" in text_a
    assert "B-一号" in text_b

    a_second_prompt = "\n".join(
        str(message.get("content")) for message in provider.calls[2]
    )
    assert "A-一号" in a_second_prompt
    assert "B-一号" not in a_second_prompt

    b_second_prompt = "\n".join(
        str(message.get("content")) for message in provider.calls[3]
    )
    assert "B-一号" in b_second_prompt
    assert "A-一号" not in b_second_prompt

    card_a = await repo.get_state_card(session_a.id)
    card_b = await repo.get_state_card(session_b.id)
    assert card_a is not None and card_b is not None
    assert card_a.session_id != card_b.session_id
    assert card_a.goal.startswith("A-一号")
    assert card_b.goal.startswith("B-一号")


async def test_two_sessions_concurrent_writes(settings: Settings, db: Any) -> None:
    session_a = await create_session(db, settings, "并发A")
    session_b = await create_session(db, settings, "并发B")
    provider_a = EchoProvider()
    provider_b = EchoProvider()
    loop_a, _ = build_loop(settings, provider_a)
    loop_b, _ = build_loop(settings, provider_b)

    await asyncio.gather(
        loop_a.run(session_a.id, "CONC-A 独立任务", Collector()),
        loop_b.run(session_b.id, "CONC-B 独立任务", Collector()),
    )

    repo = SessionRepository(db, settings.dev_user_id)
    text_a = "\n".join(
        message.content for message in await repo.list_messages(session_a.id)
    )
    text_b = "\n".join(
        message.content for message in await repo.list_messages(session_b.id)
    )
    assert "CONC-A" in text_a
    assert "CONC-B" not in text_a
    assert "CONC-B" in text_b
    assert "CONC-A" not in text_b

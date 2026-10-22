from __future__ import annotations

import os

import pytest

from app.config import Settings
from tests.helpers import Collector

pytestmark = pytest.mark.e2e


def _api_key() -> str:
    return os.environ.get("MODEL_L1_API_KEY") or os.environ.get("DEEPSEEK_API_KEY", "")


def _e2e_settings() -> Settings:
    key = _api_key()
    return Settings(
        app_env="e2e",
        database_url=os.environ.get(
            "TEST_DATABASE_URL",
            "postgresql+asyncpg://postgres:postgres@127.0.0.1:55433/agent_test",
        ),
        auto_migrate=False,
        model_l1_base_url=os.environ.get("MODEL_L1_BASE_URL", "https://api.deepseek.com"),
        model_l1_api_key=key,
        model_l1_name=os.environ.get("MODEL_L1_NAME", "deepseek-chat"),
        model_l2_base_url="",
        model_l2_api_key="",
        model_l3_base_url="",
        deepseek_search_api_key=os.environ.get("DEEPSEEK_SEARCH_API_KEY", key),
        deepseek_search_base_url=os.environ.get(
            "DEEPSEEK_SEARCH_BASE_URL", "https://api.deepseek.com"
        ),
        context_compress_threshold=100_000,
    )


@pytest.mark.skipif(not _api_key(), reason="需要真实 DeepSeek Key 才能运行联网 e2e")
async def test_real_agent_weather_and_todo(settings: Settings, db: object) -> None:
    from sqlalchemy import func, select

    from app.agent.loop import AgentLoop
    from app.config import BASE_DIR
    from app.mcp.client import MCPClient
    from app.models.db_models import Todo
    from app.models.gateway import ModelGateway
    from app.observability.tracing import get_tracer
    from app.sessions.repository import SessionRepository

    e2e_settings = _e2e_settings()
    mcp = MCPClient(e2e_settings.mcp_args, env=e2e_settings.mcp_env(), cwd=BASE_DIR, timeout=30)
    await mcp.start()
    try:
        gateway = ModelGateway(e2e_settings, tracer=get_tracer(e2e_settings))
        loop = AgentLoop(
            e2e_settings, gateway, mcp, user_id=e2e_settings.dev_user_id
        )
        repo = SessionRepository(db, e2e_settings.dev_user_id)  # type: ignore[arg-type]
        session = await repo.create(title="e2e")
        await db.commit()  # type: ignore[attr-defined]

        collector = Collector()
        result = await loop.run(
            session.id,
            "查一下今天杭州天气，并帮我建一个待办：晚上复盘今天的开发。",
            collector,
        )
        assert result.finish_reason == "final_answer"
        assert result.final_answer.strip()
        names = [data.get("name") for event, data in collector.events if event == "tool_start"]
        assert "weather" in names
        assert "todo" in names
        assert all(
            data.get("status") == "ok"
            for event, data in collector.events
            if event == "tool_result"
        )
        count = (
            await db.execute(  # type: ignore[attr-defined]
                select(func.count())
                .select_from(Todo)
                .where(Todo.user_id == e2e_settings.dev_user_id)
            )
        ).scalar_one()
        assert count >= 1
    finally:
        await mcp.close()


@pytest.mark.skipif(not _api_key(), reason="需要真实 DeepSeek Key 才能运行联网搜索 e2e")
async def test_real_deepseek_search_returns_links() -> None:
    from urllib.parse import urlparse

    from app.models.search import DeepSeekWebSearch

    provider = DeepSeekWebSearch()
    results = await provider.search("杭州 今天 天气", max_results=3)
    assert results
    assert all(item["title"] for item in results)
    assert all(item["url"].startswith("http") for item in results)
    assert all(urlparse(item["url"]).netloc for item in results)


@pytest.mark.skipif(not _api_key(), reason="需要真实 DeepSeek Key 才能运行联网搜索 e2e")
async def test_real_search_tool_via_mcp(settings: Settings) -> None:
    import json

    from app.config import BASE_DIR
    from app.mcp.client import MCPClient

    e2e_settings = _e2e_settings()
    mcp = MCPClient(
        e2e_settings.mcp_args,
        env=e2e_settings.mcp_env(),
        cwd=BASE_DIR,
        timeout=60,
    )
    await mcp.start()
    try:
        response = await mcp.call_tool(
            "search", {"query": "杭州 今天 天气", "max_results": 3}, user_id="dev-user"
        )
        assert response["isError"] is False
        payload = json.loads(response["content"][0]["text"])
        assert payload["source"] == "deepseek_web_search"
        assert payload["results"]
        assert all(
            item["url"].startswith("http") for item in payload["results"]
        )
    finally:
        await mcp.close()

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy import text

TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql+asyncpg://postgres:postgres@127.0.0.1:55433/agent_test",
)
os.environ["DATABASE_URL"] = TEST_DATABASE_URL

from app.config import BASE_DIR, Settings, reset_settings_cache  # noqa: E402
from app.db import create_all, get_engine, get_sessionmaker, reset_engine  # noqa: E402

TRUNCATE_SQL = """
TRUNCATE TABLE events, artifacts, todo_operations, tool_calls, messages, summaries,
state_cards, todos, snapshots, vector_documents, sessions, users RESTART IDENTITY CASCADE
"""


@pytest.fixture(scope="session")
def settings() -> Settings:
    reset_settings_cache()
    return Settings(
        app_env="test",
        database_url=TEST_DATABASE_URL,
        auto_migrate=False,
        docs_root=BASE_DIR / "tests" / "fixtures" / "docs",
        artifacts_dir=BASE_DIR / "tests" / ".artifacts",
        log_dir=BASE_DIR / "tests" / ".logs",
        context_compress_threshold=2000,
        llm_retry=3,
        llm_retry_base_delay=0,
        tool_retry=1,
        breaker_open_seconds=30,
        max_task_turns=30,
        max_tool_iterations=8,
        max_model_calls_per_turn=12,
    )


@pytest_asyncio.fixture(autouse=True)
async def clean_database(
    request: pytest.FixtureRequest, settings: Settings
) -> AsyncIterator[None]:
    node_id = request.node.nodeid.replace("\\", "/")
    if "/integration/" not in node_id and "/e2e/" not in node_id:
        yield
        return
    await reset_engine()
    await create_all(settings)
    engine = get_engine(settings)
    async with engine.begin() as conn:
        await conn.execute(text(TRUNCATE_SQL))
    yield
    await reset_engine()


@pytest_asyncio.fixture
async def db(settings: Settings) -> AsyncIterator[object]:
    sessionmaker = get_sessionmaker(settings)
    async with sessionmaker() as session:
        yield session


@pytest.fixture(scope="session")
def docs_fixture_dir() -> Path:
    return Path(__file__).resolve().parent / "fixtures" / "docs"

from __future__ import annotations

import logging

from sqlalchemy import inspect, text

from app.config import Settings
from app.db import get_engine, run_migrations
from app.observability import configure_logging


async def test_alembic_upgrade_from_empty_schema_keeps_logger(settings: Settings) -> None:
    configure_logging(settings)
    logger = logging.getLogger("agent")
    assert logger.disabled is False
    assert logger.handlers

    engine = get_engine(settings)
    async with engine.begin() as conn:
        await conn.execute(text("DROP SCHEMA public CASCADE"))
        await conn.execute(text("CREATE SCHEMA public"))

    await run_migrations(settings)

    async with engine.connect() as conn:
        tables = await conn.run_sync(lambda sync_conn: inspect(sync_conn).get_table_names())
    assert {"alembic_version", "sessions", "messages", "events", "todos"} <= set(tables)
    assert logger.disabled is False
    assert logger.handlers

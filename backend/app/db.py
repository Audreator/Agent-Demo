from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.config import BASE_DIR, Settings, get_settings


class Base(DeclarativeBase):
    pass


_engine: AsyncEngine | None = None
_sessionmaker: async_sessionmaker[AsyncSession] | None = None
_engine_url: str | None = None


def get_engine(settings: Settings | None = None) -> AsyncEngine:
    global _engine, _engine_url, _sessionmaker
    settings = settings or get_settings()
    if _engine is None or _engine_url != settings.database_url:
        _engine = create_async_engine(
            settings.database_url,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
        )
        _engine_url = settings.database_url
        _sessionmaker = async_sessionmaker(_engine, expire_on_commit=False)
    return _engine


def get_sessionmaker(settings: Settings | None = None) -> async_sessionmaker[AsyncSession]:
    get_engine(settings)
    assert _sessionmaker is not None
    return _sessionmaker


async def reset_engine() -> None:
    global _engine, _sessionmaker, _engine_url
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _sessionmaker = None
    _engine_url = None


async def create_all(settings: Settings | None = None) -> None:
    from app.models.db_models import Base as ModelsBase  # noqa: F401  ensure models imported

    engine = get_engine(settings)
    async with engine.begin() as conn:
        await conn.run_sync(ModelsBase.metadata.create_all)


def _upgrade_sync(settings: Settings) -> None:
    from alembic.config import Config

    from alembic import command

    config = Config(str(BASE_DIR / "alembic.ini"))
    config.set_main_option("script_location", str(BASE_DIR / "alembic"))
    config.set_main_option("sqlalchemy.url", settings.database_url)
    command.upgrade(config, "head")


async def run_migrations(settings: Settings | None = None) -> None:
    await asyncio.to_thread(_upgrade_sync, settings or get_settings())


async def get_db() -> AsyncIterator[AsyncSession]:
    sessionmaker = get_sessionmaker()
    async with sessionmaker() as session:
        yield session


@asynccontextmanager
async def session_scope(settings: Settings | None = None) -> AsyncIterator[AsyncSession]:
    sessionmaker = get_sessionmaker(settings)
    async with sessionmaker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise

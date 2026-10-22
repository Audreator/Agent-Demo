from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.agent.loop import AgentLoop
from app.api.routes import router
from app.api.sse import StreamRegistry
from app.config import BASE_DIR, Settings, get_settings
from app.db import get_sessionmaker, run_migrations
from app.mcp.client import MCPClient
from app.models.db_models import User
from app.models.gateway import ModelGateway
from app.observability import configure_logging, get_logger


@dataclass
class AppState:
    settings: Settings
    gateway: ModelGateway
    mcp: MCPClient
    loop: AgentLoop
    streams: StreamRegistry
    tools: list[dict[str, Any]]
    session_locks: dict[str, asyncio.Lock] = field(default_factory=dict)


def create_app(
    settings: Settings | None = None,
    *,
    gateway: ModelGateway | None = None,
    mcp: MCPClient | None = None,
    auto_start_mcp: bool = True,
) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings)
    logger = get_logger()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if settings.auto_migrate:
            await run_migrations(settings)
            await _ensure_dev_user(settings)
        gateway_instance = gateway or ModelGateway(settings)
        mcp_instance = mcp or MCPClient(
            settings.mcp_args,
            env=settings.mcp_env(),
            cwd=BASE_DIR,
            timeout=settings.mcp_timeout,
        )
        tools: list[dict[str, Any]] = []
        if auto_start_mcp:
            try:
                await mcp_instance.start()
                tools = await mcp_instance.list_tools()
            except Exception as exc:
                logger.warning("mcp_start_failed", extra={"payload": {"error": str(exc)}})
        loop = AgentLoop(settings, gateway_instance, mcp_instance, user_id=settings.dev_user_id)
        app.state.app_state = AppState(
            settings=settings,
            gateway=gateway_instance,
            mcp=mcp_instance,
            loop=loop,
            streams=StreamRegistry(),
            tools=tools,
            session_locks={},
        )
        logger.info(
            "app_started",
            extra={
                "payload": {
                    "env": settings.app_env,
                    "models": gateway_instance.status(),
                    "tools": [tool.get("name") for tool in tools],
                }
            },
        )
        try:
            yield
        finally:
            await mcp_instance.close()

    app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router, prefix="/api")
    return app


async def _ensure_dev_user(settings: Settings) -> None:
    sessionmaker = get_sessionmaker(settings)
    async with sessionmaker() as db:
        user = await db.get(User, settings.dev_user_id)
        if user is None:
            db.add(User(id=settings.dev_user_id, name=settings.dev_user_id))
            await db.commit()


app = create_app()


def main() -> None:  # pragma: no cover
    settings = get_settings()
    uvicorn.run(
        "app.main:app", host=settings.host, port=settings.port, reload=settings.app_env == "dev"
    )


__all__ = ["AppState", "app", "create_app", "main"]

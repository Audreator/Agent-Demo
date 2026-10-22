from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(BASE_DIR / ".env", BASE_DIR.parent / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_env: str = "dev"
    app_name: str = "minimal-agent"
    host: str = "127.0.0.1"
    port: int = 8000
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    database_url: str = "postgresql+asyncpg://postgres:postgres@127.0.0.1:55432/agent"
    auto_migrate: bool = True

    docs_root: Path = BASE_DIR / "docs"
    agent_md_path: Path | None = None
    artifacts_dir: Path = BASE_DIR / "artifacts"
    log_dir: Path = BASE_DIR / "logs"
    dev_user_id: str = "dev-user"

    context_window: int = 1_000_000
    context_budget_ratio: float = 0.7
    context_compress_threshold: int | None = None
    keep_recent_turns: int = 5
    max_task_turns: int = 30
    max_tool_iterations: int = 8
    max_model_calls_per_turn: int = 12
    max_self_correct: int = 5
    tool_retry: int = 3
    llm_retry: int = 3
    llm_retry_base_delay: float = 0.5
    breaker_open_seconds: int = 30
    tool_result_context_limit: int = 2000
    search_result_limit: int = 8000
    max_output_tokens: int = 4096

    model_layers: str = "deepseek,qwen,local"
    model_l1_base_url: str = ""
    model_l1_api_key: str = ""
    model_l1_name: str = "deepseek-chat"
    model_l2_base_url: str = ""
    model_l2_api_key: str = ""
    model_l2_name: str = "qwen-plus"
    model_l3_base_url: str = ""
    model_l3_api_key: str = ""
    model_l3_name: str = "local-model"
    model_timeout: float = 60.0

    deepseek_search_enabled: bool = True
    deepseek_search_base_url: str = "https://api.deepseek.com"
    deepseek_search_api_key: str = ""
    deepseek_search_model: str = "deepseek-v4-pro"

    mcp_command: str = ""
    mcp_timeout: float = 30.0

    @field_validator("agent_md_path", mode="before")
    @classmethod
    def _empty_path_to_none(cls, value: object) -> object:
        if value == "" or value is None:
            return None
        return value

    @field_validator("docs_root", "artifacts_dir", "log_dir", mode="after")
    @classmethod
    def _absolute_path(cls, value: Path) -> Path:
        return value if value.is_absolute() else (BASE_DIR / value).resolve()

    @field_validator("agent_md_path", mode="after")
    @classmethod
    def _absolute_agent_md(cls, value: Path | None) -> Path | None:
        if value is None or value.is_absolute():
            return value
        return (BASE_DIR / value).resolve()

    @property
    def threshold(self) -> int:
        if self.context_compress_threshold:
            return self.context_compress_threshold
        return int(self.context_window * self.context_budget_ratio)

    @property
    def cors_origin_list(self) -> list[str]:
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]

    @property
    def layer_order(self) -> list[str]:
        return [item.strip() for item in self.model_layers.split(",") if item.strip()]

    @property
    def agent_md_file(self) -> Path:
        if self.agent_md_path is not None:
            return self.agent_md_path
        return BASE_DIR / "Agent.md"

    @property
    def mcp_args(self) -> list[str]:
        if self.mcp_command.strip():
            return self.mcp_command.split()
        # The server module is launched with the same interpreter as the web app.
        import sys

        return [sys.executable, "-m", "app.mcp.server"]

    def mcp_env(self) -> dict[str, str]:
        import os

        env = dict(os.environ)
        env["DATABASE_URL"] = self.database_url
        env["DOCS_ROOT"] = str(self.docs_root)
        env["ARTIFACTS_DIR"] = str(self.artifacts_dir)
        env["SEARCH_RESULT_LIMIT"] = str(self.search_result_limit)
        env["DEEPSEEK_SEARCH_ENABLED"] = str(self.deepseek_search_enabled).lower()
        env["DEEPSEEK_SEARCH_BASE_URL"] = self.deepseek_search_base_url
        env["DEEPSEEK_SEARCH_API_KEY"] = self.deepseek_search_api_key
        env["DEEPSEEK_SEARCH_MODEL"] = self.deepseek_search_model
        env["MODEL_L1_BASE_URL"] = self.model_l1_base_url
        env["MODEL_L1_API_KEY"] = self.model_l1_api_key
        env["MODEL_L1_NAME"] = self.model_l1_name
        env.setdefault("PYTHONUNBUFFERED", "1")
        return env


@lru_cache
def get_settings() -> Settings:
    return Settings()


TraceMode = Literal["react", "native"]


def reset_settings_cache() -> None:
    get_settings.cache_clear()


__all__ = ["Settings", "get_settings", "reset_settings_cache", "BASE_DIR", "TraceMode", "Field"]

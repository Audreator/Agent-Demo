from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

from app.config import Settings, get_settings
from app.errors import (
    AgentError,
    PermanentError,
    ServiceUnavailableError,
    TransientError,
)
from app.models.breaker import Breaker
from app.models.providers import (
    ChatChunk,
    ChatProvider,
    ModelResult,
    ProviderConfig,
    merge_tool_call_deltas,
)
from app.observability.tracing import Tracer, get_tracer

DeltaCallback = Callable[[ChatChunk], Awaitable[None]]
ResetCallback = Callable[[], Awaitable[None]]


class Provider(Protocol):
    config: ProviderConfig

    @property
    def name(self) -> str: ...

    def stream(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> Any: ...


@dataclass
class GatewayProviderState:
    provider: Provider
    breaker: Breaker


def build_provider_configs(settings: Settings) -> list[ProviderConfig]:
    mapping = {
        "deepseek": (
            settings.model_l1_base_url,
            settings.model_l1_api_key,
            settings.model_l1_name,
        ),
        "qwen": (
            settings.model_l2_base_url,
            settings.model_l2_api_key,
            settings.model_l2_name,
        ),
        "local": (
            settings.model_l3_base_url,
            settings.model_l3_api_key,
            settings.model_l3_name,
        ),
    }
    configs: list[ProviderConfig] = []
    for layer in settings.layer_order:
        if layer not in mapping:
            continue
        base_url, api_key, model = mapping[layer]
        configs.append(
            ProviderConfig(
                layer=layer,
                name=layer,
                base_url=base_url,
                api_key=api_key,
                model=model,
                timeout=settings.model_timeout,
            )
        )
    return configs


class ModelGateway:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        providers: list[Provider] | None = None,
        tracer: Tracer | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        jitter: Callable[[], float] = random.random,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.tracer = tracer or get_tracer(self.settings)
        self._sleep = sleep
        self._jitter = jitter
        configs = build_provider_configs(self.settings)
        self.providers: list[Provider] = (
            providers if providers is not None else [ChatProvider(config) for config in configs]
        )
        clock_fn = clock or __import__("time").monotonic
        self.breakers: dict[str, Breaker] = {
            provider.name: Breaker(
                open_seconds=self.settings.breaker_open_seconds, clock=clock_fn
            )
            for provider in self.providers
        }

    def status(self) -> list[dict[str, Any]]:
        result = []
        for provider in self.providers:
            config = provider.config
            breaker = self.breakers.get(provider.name)
            item = config.snapshot()
            item["breaker"] = breaker.snapshot() if breaker else None
            result.append(item)
        return result

    def has_available_model(self) -> bool:
        return any(provider.config.configured for provider in self.providers)

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        on_delta: DeltaCallback | None = None,
        on_reset: ResetCallback | None = None,
        max_tokens: int | None = None,
    ) -> ModelResult:
        last_error: AgentError | None = None
        configured = sum(1 for provider in self.providers if provider.config.configured)
        for provider in self.providers:
            if not provider.config.configured:
                continue
            breaker = self.breakers[provider.name]
            if not breaker.allow():
                await self._record(
                    "breaker_skip", {"provider": provider.name, "state": breaker.state}
                )
                continue
            text_parts: list[str] = []
            reasoning_parts: list[str] = []
            tool_calls: dict[int, dict[str, Any]] = {}
            usage: dict[str, Any] = {}

            for attempt in range(max(1, self.settings.llm_retry)):
                emitted = False
                text_parts.clear()
                reasoning_parts.clear()
                tool_calls.clear()
                usage = {}
                try:
                    async with self.tracer.span(
                        "model_call",
                        provider=provider.name,
                        layer=provider.config.layer,
                        attempt=attempt + 1,
                        mode="native" if tools else "react",
                    ) as span:
                        async for chunk in provider.stream(
                            messages,
                            tools=tools,
                            max_tokens=max_tokens,
                        ):
                            if chunk.content:
                                text_parts.append(chunk.content)
                            if chunk.reasoning:
                                reasoning_parts.append(chunk.reasoning)
                            if chunk.tool_calls:
                                merge_tool_call_deltas(tool_calls, chunk.tool_calls)
                            if chunk.usage:
                                usage = chunk.usage
                            if on_delta is not None and (
                                chunk.content or chunk.reasoning or chunk.tool_calls
                            ):
                                emitted = True
                                await on_delta(chunk)
                        span.set(
                            content_chars=sum(len(part) for part in text_parts),
                            reasoning_chars=sum(len(part) for part in reasoning_parts),
                            usage=usage,
                        )
                    breaker.on_success()
                    await self._record(
                        "model_success",
                        {
                            "provider": provider.name,
                            "layer": provider.config.layer,
                            "usage": usage,
                            "content_chars": sum(len(part) for part in text_parts),
                        },
                    )
                    return ModelResult(
                        text="".join(text_parts),
                        reasoning="".join(reasoning_parts),
                        tool_calls=[tool_calls[index] for index in sorted(tool_calls)],
                        usage=usage,
                        provider=provider.name,
                        layer=provider.config.layer,
                        model=provider.config.model,
                    )
                except PermanentError as exc:
                    last_error = exc
                    if emitted and on_reset is not None:
                        await on_reset()
                    breaker.on_failure()
                    await self._record(
                        "model_permanent_failure",
                        {
                            "provider": provider.name,
                            "status_code": exc.status_code,
                            "error": exc.message,
                        },
                    )
                    break
                except TransientError as exc:
                    last_error = exc
                    await self._record(
                        "model_transient_failure",
                        {
                            "provider": provider.name,
                            "attempt": attempt + 1,
                            "chunks_started": emitted,
                            "error": exc.message,
                            "status_code": exc.status_code,
                        },
                    )
                    if emitted:
                        if on_reset is None:
                            breaker.on_failure()
                            raise ServiceUnavailableError(
                                "模型流式输出中断，请重试本轮请求"
                            ) from exc
                        await on_reset()
                        await self._record(
                            "stream_reset",
                            {"provider": provider.name, "attempt": attempt + 1},
                        )
                    if attempt < max(1, self.settings.llm_retry) - 1:
                        delay = self._retry_delay(attempt, exc)
                        await self._sleep(delay)
                        continue
                    breaker.on_failure()
                    await self._record(
                        "breaker_open", {"provider": provider.name, "reason": "retries_exhausted"}
                    )
                    break
                except asyncio.CancelledError:
                    breaker.release_probe()
                    raise
                except Exception as exc:
                    last_error = TransientError(f"{provider.name} 调用异常: {exc}")
                    if emitted and on_reset is not None:
                        await on_reset()
                    await self._record(
                        "model_unknown_failure", {"provider": provider.name, "error": str(exc)}
                    )
                    if attempt < max(1, self.settings.llm_retry) - 1:
                        await self._sleep(self._retry_delay(attempt, last_error))
                        continue
                    breaker.on_failure()
                    break
            await self._record(
                "model_degrade", {"from": provider.name, "reason": str(last_error)}
            )

        if configured == 0:
            raise ServiceUnavailableError(
                "未配置任何可用模型层，请检查 MODEL_L1/L2/L3 环境变量"
            )
        raise ServiceUnavailableError("服务暂不可用，请稍后重试")

    def _retry_delay(self, attempt: int, error: AgentError) -> float:
        base = max(0.0, self.settings.llm_retry_base_delay) * (2**attempt)
        delay = base + base * 0.25 * self._jitter()
        retry_after = error.data.get("retry_after")
        if retry_after is not None:
            try:
                delay = max(delay, float(str(retry_after)))
            except (TypeError, ValueError):
                try:
                    from email.utils import parsedate_to_datetime

                    when = parsedate_to_datetime(str(retry_after))
                    now = datetime.now(UTC)
                    delay = max(delay, max(0.0, (when - now).total_seconds()))
                except (TypeError, ValueError, OverflowError):
                    pass
        return delay

    async def _record(self, kind: str, payload: dict[str, Any]) -> None:
        try:
            await self.tracer.emit_event(kind, payload)
        except Exception:
            pass


__all__ = [
    "GatewayProviderState",
    "ModelGateway",
    "Provider",
    "build_provider_configs",
]

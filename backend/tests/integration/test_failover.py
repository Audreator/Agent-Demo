from __future__ import annotations

import asyncio

import pytest

from app.config import Settings
from app.errors import PermanentError, ServiceUnavailableError, TransientError
from app.models.gateway import ModelGateway
from app.models.providers import ChatChunk, ProviderConfig
from app.observability.tracing import get_tracer
from tests.helpers import ScriptedProvider


async def _no_sleep(_: float) -> None:
    await asyncio.sleep(0)


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def make_gateway(settings: Settings, providers, clock=None) -> ModelGateway:
    return ModelGateway(
        settings,
        providers=providers,
        tracer=get_tracer(settings),
        sleep=_no_sleep,
        jitter=lambda: 0.0,
        clock=clock or (lambda: 1000.0),
    )


async def test_retry_then_success_same_layer(settings: Settings) -> None:
    provider = ScriptedProvider(
        scripts=[TransientError("429"), "Thought: ok\nFinal Answer: 成功"]
    )
    gateway = make_gateway(settings, [provider])
    result = await gateway.complete([{"role": "user", "content": "hi"}])
    assert result.text.endswith("成功")
    assert len(provider.calls) == 2
    assert gateway.breakers[provider.name].state == "closed"


async def test_three_layers_fail_and_half_open_probe(settings: Settings) -> None:
    clock = FakeClock()
    p1 = ScriptedProvider(scripts=[TransientError("down1") for _ in range(3)], name="l1")
    p2 = ScriptedProvider(
        scripts=[TransientError("down2") for _ in range(3)], name="l2", layer="qwen"
    )
    p3 = ScriptedProvider(
        scripts=[TransientError("down3") for _ in range(3)], name="l3", layer="local"
    )
    gateway = make_gateway(settings, [p1, p2, p3], clock=clock)

    with pytest.raises(ServiceUnavailableError, match="稍后重试"):
        await gateway.complete([{"role": "user", "content": "hi"}])

    assert len(p1.calls) == 3
    assert len(p2.calls) == 3
    assert len(p3.calls) == 3
    assert all(gateway.breakers[name].state == "open" for name in ("l1", "l2", "l3"))

    with pytest.raises(ServiceUnavailableError):
        await gateway.complete([{"role": "user", "content": "still down"}])
    assert len(p1.calls) == 3

    clock.now += 31
    p1.scripts.append("Thought: 恢复\nFinal Answer: 已恢复")
    result = await gateway.complete([{"role": "user", "content": "recover"}])
    assert result.provider == "l1"
    assert gateway.breakers["l1"].state == "closed"
    assert p1.calls


async def test_permanent_error_does_not_retry_and_degrades(settings: Settings) -> None:
    p1 = ScriptedProvider(scripts=[PermanentError("401", status_code=401)], name="l1")
    p2 = ScriptedProvider(scripts=["Thought: ok\nFinal Answer: 第二层"], name="l2", layer="qwen")
    gateway = make_gateway(settings, [p1, p2])
    result = await gateway.complete([{"role": "user", "content": "hi"}])
    assert result.provider == "l2"
    assert len(p1.calls) == 1
    assert gateway.breakers["l1"].state == "open"


class MidStreamFailingProvider:
    def __init__(self, name: str) -> None:
        self.calls = 0
        self.config = ProviderConfig(
            layer="deepseek",
            name=name,
            base_url="http://fake.local/v1",
            api_key="test-key",
            model="fake",
        )

    @property
    def name(self) -> str:
        return self.config.name

    async def stream(self, messages, *, tools=None, temperature=0.2, max_tokens=None):
        self.calls += 1
        yield ChatChunk(content="半截输出")
        raise TransientError("stream reset")


async def test_midstream_failure_fails_over_without_duplicate_deltas(
    settings: Settings,
) -> None:
    broken = MidStreamFailingProvider("l1")
    healthy = ScriptedProvider(
        scripts=["完整答案"], name="l2", layer="qwen"
    )
    gateway = make_gateway(settings, [broken, healthy])
    deltas: list[str] = []
    resets = 0

    async def on_delta(chunk: ChatChunk) -> None:
        if chunk.content:
            deltas.append(chunk.content)

    async def on_reset() -> None:
        nonlocal resets
        resets += 1

    result = await gateway.complete(
        [{"role": "user", "content": "hi"}], on_delta=on_delta, on_reset=on_reset
    )
    assert result.provider == "l2"
    assert broken.calls == 3
    assert resets >= 1
    assert "".join(deltas).endswith("完整答案")
    assert gateway.breakers["l1"].state == "open"

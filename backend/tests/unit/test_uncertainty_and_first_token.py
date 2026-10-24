from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any

import pytest

from app.admin.service import first_token_value, percentile
from app.agent.loop import AgentLoop, StreamSplitter
from app.agent.uncertainty import (
    forced_search_query,
    is_refusal_answer,
    should_force_search,
)
from app.config import Settings
from app.errors import PermanentError, ServiceUnavailableError, TransientError
from app.models.gateway import ModelGateway
from app.models.providers import ChatChunk, ProviderConfig
from app.observability.tracing import Tracer
from tests.helpers import Collector


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("我不知道这个问题的答案。", True),
        ("不知道。", True),
        ("不清楚这个数据。", True),
        ("我不确定。", True),
        ("我没找到相关资料。", True),
        ("很抱歉，我无法确定。", True),
        ("I'm not sure about that.", True),
        ("I’m not sure.", True),
        ("无法提供相关信息。", True),
        ("根据检索结果，2025 年营收为 120 亿元。https://example.com", False),
        ("我不知道，但详情见 https://example.com", False),
        ("很抱歉，根据检索结果，2026 年人口为 42。", False),
        ("不确定的部分我已经在第三段说明了全部结论。", False),
        ("长" * 300 + "我不知道", False),
        ("长" * 900 + "我不知道", False),
        ("", False),
    ],
)
def test_is_refusal_answer(answer: str, expected: bool) -> None:
    assert is_refusal_answer(answer) is expected


def test_refusal_match_window_and_length_boundaries() -> None:
    assert is_refusal_answer("x" * 199 + "我不知道") is True
    assert is_refusal_answer("x" * 200 + "我不知道") is False
    assert is_refusal_answer("我不知道" + "x" * 796) is True
    assert is_refusal_answer("我不知道" + "x" * 797) is False


def test_forced_search_query_strips_and_truncates() -> None:
    assert forced_search_query("  火星人口  ", 500) == "火星人口"
    assert len(forced_search_query("x" * 600, 500)) == 500
    assert forced_search_query("", 500) == ""
    assert forced_search_query(None, 500) == ""  # type: ignore[arg-type]


def test_should_force_search_conditions() -> None:
    base = {
        "answer": "我不知道",
        "has_ok_search": False,
        "already_forced": False,
        "enabled": True,
        "finish_kind": "final",
    }
    assert should_force_search(**base) is True
    assert should_force_search(**{**base, "has_ok_search": True}) is False
    assert should_force_search(**{**base, "already_forced": True}) is False
    assert should_force_search(**{**base, "enabled": False}) is False
    assert should_force_search(**{**base, "finish_kind": "wrap_up"}) is False
    assert should_force_search(**{**base, "finish_kind": "budget_exceeded"}) is False


async def test_react_final_answer_is_buffered_while_thought_streams() -> None:
    collector = Collector()
    splitter = StreamSplitter("react", collector, buffer_limit=100)
    await splitter.feed(ChatChunk(content="Thought: 先想想\nFinal Answer: 我"))
    await splitter.feed(ChatChunk(content="不知道"))

    names = collector.names()
    assert "thought_delta" in names
    assert "message_delta" not in names
    assert splitter.take_final().strip() == "我不知道"

    await splitter.flush()
    assert "message_delta" not in collector.names()


async def test_native_content_is_buffered_and_reasoning_streams() -> None:
    collector = Collector()
    splitter = StreamSplitter("native", collector)
    await splitter.feed(ChatChunk(content="我不知道", reasoning="思考中"))

    names = collector.names()
    assert names.count("thought_delta") == 1
    assert "message_delta" not in names
    assert splitter.take_final() == "我不知道"
    assert splitter.buffer_limit_hit is False


async def test_buffer_limit_releases_immediately_and_skips_decision() -> None:
    collector = Collector()
    splitter = StreamSplitter("native", collector, buffer_limit=5)
    await splitter.feed(ChatChunk(content="123456"))
    assert splitter.buffer_limit_hit is True
    assert splitter.take_final() == ""
    assert collector.first("message_delta") == {"delta": "123456"}
    await splitter.feed(ChatChunk(content="789"))
    assert collector.first("message_delta") == {"delta": "123456"}
    assert collector.events[-1][0] == "message_delta"


async def test_reset_clears_buffer_and_limit_flag() -> None:
    collector = Collector()
    splitter = StreamSplitter("native", collector, buffer_limit=5)
    await splitter.feed(ChatChunk(content="123456"))
    await splitter.reset()
    assert splitter.take_final() == ""
    assert splitter.buffer_limit_hit is False
    assert collector.events[-1] == ("context_update", {"reset": True})


def test_percentile_and_first_token_value_handle_empty_samples() -> None:
    assert percentile([], 95) is None
    assert percentile([100, 200, 300], 95) == 300.0
    assert percentile([0, 0], 95) == 0.0
    assert first_token_value({}) is None
    assert first_token_value(None) is None
    assert first_token_value({"first_token_ms": None}) is None
    assert first_token_value({"first_token_ms": -1}) is None
    assert first_token_value({"first_token_ms": 0}) == 0
    assert first_token_value({"first_token_ms": 123}) == 123


class FakeSpan:
    def __init__(self) -> None:
        self.fields: dict[str, Any] = {}

    def set(self, **values: Any) -> None:
        self.fields.update(values)


async def test_turn_ttft_only_written_on_first_visible_delta() -> None:
    collector = Collector()
    span = FakeSpan()
    emit = AgentLoop._with_turn_ttft(collector, span, time.monotonic())

    await emit("meta", {"turn_no": 1})
    await emit("tool_start", {"name": "search"})
    assert "first_token_ms" not in span.fields

    await emit("message_delta", {"delta": "回答"})
    assert span.fields["first_token_ms"] >= 0
    first = span.fields["first_token_ms"]
    await emit("message_delta", {"delta": "继续"})
    assert span.fields["first_token_ms"] == first
    assert collector.names() == ["meta", "tool_start", "message_delta", "message_delta"]


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class RecordingTracer(Tracer):
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.events: list[tuple[str, dict[str, Any]]] = []

    async def emit_event(self, kind: str, payload: dict[str, Any] | None = None, **_: Any) -> None:
        self.events.append((kind, payload or {}))

    def payloads(self, kind: str) -> list[dict[str, Any]]:
        return [payload for name, payload in self.events if name == kind]


class TimedProvider:
    def __init__(self, clock: FakeClock, steps: list[tuple[float, Any]]) -> None:
        self.clock = clock
        self.steps = steps
        self.calls = 0
        self.config = ProviderConfig(
            layer="deepseek",
            name="timed",
            base_url="http://fake.local/v1",
            api_key="test-key",
            model="timed-model",
        )

    @property
    def name(self) -> str:
        return self.config.name

    async def stream(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> AsyncIterator[ChatChunk]:
        self.calls += 1
        for delay, step in self.steps:
            self.clock.now += delay
            if isinstance(step, Exception):
                raise step
            yield step


async def _no_sleep(_: float) -> None:
    await asyncio.sleep(0)


def build_gateway(
    settings: Settings, provider: TimedProvider, tracer: RecordingTracer, clock: FakeClock
) -> ModelGateway:
    return ModelGateway(
        settings,
        providers=[provider],
        tracer=tracer,
        sleep=_no_sleep,
        jitter=lambda: 0.0,
        clock=clock,
    )


async def test_first_token_skips_empty_chunks(settings: Settings) -> None:
    clock = FakeClock()
    tracer = RecordingTracer(settings)
    provider = TimedProvider(
        clock,
        [
            (0.05, ChatChunk(content="")),
            (0.06, ChatChunk(content="最终答案")),
        ],
    )
    gateway = build_gateway(settings, provider, tracer, clock)
    await gateway.complete([{"role": "user", "content": "hi"}])

    success = tracer.payloads("model_success")
    assert len(success) == 1
    assert success[0]["first_token_ms"] == pytest.approx(110, abs=1)
    calls = tracer.payloads("model_call")
    assert calls and calls[0]["first_token_ms"] == pytest.approx(110, abs=1)


async def test_first_token_only_counts_final_successful_attempt(settings: Settings) -> None:
    clock = FakeClock()
    tracer = RecordingTracer(settings)

    class FlakyProvider(TimedProvider):
        async def stream(
            self,
            messages: list[dict[str, Any]],
            *,
            tools: list[dict[str, Any]] | None = None,
            temperature: float = 0.2,
            max_tokens: int | None = None,
        ) -> AsyncIterator[ChatChunk]:
            self.calls += 1
            if self.calls == 1:
                clock.now += 0.05
                yield ChatChunk(content="半截")
                raise TransientError("stream reset")
            clock.now += 0.02
            yield ChatChunk(content="完整答案")

    provider = FlakyProvider(clock, [])
    gateway = build_gateway(settings, provider, tracer, clock)
    resets: list[int] = []

    async def on_delta(_: ChatChunk) -> None:
        return None

    async def on_reset() -> None:
        resets.append(1)

    result = await gateway.complete(
        [{"role": "user", "content": "hi"}], on_delta=on_delta, on_reset=on_reset
    )
    assert result.text == "完整答案"
    assert resets
    success = tracer.payloads("model_success")
    assert len(success) == 1
    assert success[0]["first_token_ms"] == pytest.approx(20, abs=1)


async def test_failed_attempt_records_null_first_token(settings: Settings) -> None:
    clock = FakeClock()
    tracer = RecordingTracer(settings)

    class FailingProvider(TimedProvider):
        async def stream(
            self,
            messages: list[dict[str, Any]],
            *,
            tools: list[dict[str, Any]] | None = None,
            temperature: float = 0.2,
            max_tokens: int | None = None,
        ) -> AsyncIterator[ChatChunk]:
            self.calls += 1
            raise PermanentError("401", status_code=401)
            yield ChatChunk()  # pragma: no cover

    provider = FailingProvider(clock, [])
    gateway = build_gateway(settings, provider, tracer, clock)
    with pytest.raises(ServiceUnavailableError):
        await gateway.complete([{"role": "user", "content": "hi"}])

    calls = tracer.payloads("model_call")
    assert calls
    assert "first_token_ms" in calls[0]
    assert calls[0]["first_token_ms"] is None

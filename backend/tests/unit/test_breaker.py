from __future__ import annotations

from app.models.breaker import Breaker


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_breaker_opens_after_failures_and_half_open_after_timeout() -> None:
    clock = FakeClock()
    breaker = Breaker(open_seconds=30, clock=clock)
    assert breaker.allow() is True
    breaker.on_failure()
    assert breaker.state == "open"
    assert breaker.allow() is False

    clock.now += 29
    assert breaker.allow() is False

    clock.now += 2
    assert breaker.allow() is True
    assert breaker.state == "half_open"
    assert breaker.probe_inflight is True
    assert breaker.allow() is False

    breaker.on_success()
    assert breaker.state == "closed"
    assert breaker.probe_inflight is False


def test_half_open_failure_reopens() -> None:
    clock = FakeClock()
    breaker = Breaker(open_seconds=30, clock=clock)
    breaker.on_failure()
    clock.now += 31
    assert breaker.allow() is True
    breaker.on_failure()
    assert breaker.state == "open"
    assert breaker.allow() is False
    clock.now += 31
    assert breaker.allow() is True

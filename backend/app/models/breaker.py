from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

BreakerState = Literal["closed", "open", "half_open"]


@dataclass
class Breaker:
    open_seconds: float = 30.0
    state: BreakerState = "closed"
    opened_at: float = 0.0
    probe_inflight: bool = False
    failure_count: int = 0
    success_count: int = 0
    clock: Callable[[], float] = field(default=time.monotonic)

    def allow(self) -> bool:
        if self.state == "closed":
            return True
        if self.state == "open":
            if self.clock() - self.opened_at >= self.open_seconds:
                self.state = "half_open"
                self.probe_inflight = False
            else:
                return False
        if self.state == "half_open":
            if self.probe_inflight:
                return False
            self.probe_inflight = True
            return True
        return False

    def release_probe(self) -> None:
        if self.state == "half_open":
            self.probe_inflight = False

    def on_success(self) -> None:
        self.state = "closed"
        self.probe_inflight = False
        self.failure_count = 0
        self.success_count += 1

    def on_failure(self) -> None:
        self.state = "open"
        self.opened_at = self.clock()
        self.probe_inflight = False
        self.failure_count += 1

    def snapshot(self) -> dict[str, object]:
        return {
            "state": self.state,
            "opened_at": self.opened_at,
            "probe_inflight": self.probe_inflight,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
        }


__all__ = ["Breaker", "BreakerState"]

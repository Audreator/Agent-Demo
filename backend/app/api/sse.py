from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID


def encode_sse(event: str, payload: dict[str, Any]) -> str:
    seq = payload.get("seq")
    lines = []
    if seq is not None:
        lines.append(f"id: {seq}")
    lines.append(f"event: {event}")
    lines.append(f"data: {json.dumps(payload, ensure_ascii=False, default=str)}")
    return "\n".join(lines) + "\n\n"


@dataclass
class StreamBuffer:
    session_id: UUID
    request_text: str = ""
    client_message_id: str | None = None
    entries: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    completed: bool = False
    created_at: float = field(default_factory=time.time)
    changed: asyncio.Event = field(default_factory=asyncio.Event)
    task: asyncio.Task[Any] | None = None

    def append(self, event: str, payload: dict[str, Any]) -> None:
        self.entries.append((event, payload))
        self.changed.set()

    def finish(self) -> None:
        self.completed = True
        self.changed.set()


class StreamRegistry:
    def __init__(self, max_sessions: int = 50) -> None:
        self._streams: dict[str, StreamBuffer] = {}
        self._max_sessions = max_sessions

    def create(
        self,
        session_id: UUID,
        request_text: str = "",
        client_message_id: str | None = None,
    ) -> StreamBuffer:
        buffer = StreamBuffer(
            session_id=session_id,
            request_text=request_text,
            client_message_id=client_message_id,
        )
        self._streams[str(session_id)] = buffer
        self._prune()
        return buffer

    def get(self, session_id: UUID) -> StreamBuffer | None:
        return self._streams.get(str(session_id))

    def remove(self, session_id: UUID) -> None:
        self._streams.pop(str(session_id), None)

    def _prune(self) -> None:
        if len(self._streams) <= self._max_sessions:
            return
        completed = sorted(
            (
                (key, buffer)
                for key, buffer in self._streams.items()
                if buffer.completed
            ),
            key=lambda item: item[1].created_at,
        )
        for key, _ in completed[: len(self._streams) - self._max_sessions]:
            self._streams.pop(key, None)


class SSEEmitter:
    def __init__(self, buffer: StreamBuffer) -> None:
        self.buffer = buffer
        self.seq = 0

    async def emit(self, event: str, data: dict[str, Any]) -> None:
        self.seq += 1
        payload = {"seq": self.seq, **data}
        self.buffer.append(event, payload)


def replay_events(buffer: StreamBuffer, last_event_id: int) -> list[tuple[str, dict[str, Any]]]:
    return [
        (event, payload)
        for event, payload in buffer.entries
        if int(payload.get("seq", 0)) > last_event_id
    ]


async def stream_buffer(
    buffer: StreamBuffer, start_after_seq: int = 0
) -> AsyncIterator[str]:
    """Yield events after the given seq, then follow the live buffer until done."""
    index = 0
    while (
        index < len(buffer.entries)
        and int(buffer.entries[index][1].get("seq", 0)) <= start_after_seq
    ):
        index += 1
    while True:
        buffer.changed.clear()
        while index < len(buffer.entries):
            event, payload = buffer.entries[index]
            index += 1
            yield encode_sse(event, payload)
        if buffer.completed and index >= len(buffer.entries):
            break
        await buffer.changed.wait()


__all__ = [
    "SSEEmitter",
    "StreamBuffer",
    "StreamRegistry",
    "encode_sse",
    "replay_events",
    "stream_buffer",
]

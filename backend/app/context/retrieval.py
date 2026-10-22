from __future__ import annotations

from typing import Any, Protocol
from uuid import UUID


class VectorRetriever(Protocol):
    """P1 reserved interface. No embedding pipeline is built in this phase."""

    async def index(
        self, session_id: UUID, content: str, metadata: dict[str, Any]
    ) -> None: ...

    async def search(
        self, session_id: UUID, query: str, limit: int = 5
    ) -> list[dict[str, Any]]: ...


class DisabledVectorRetriever:
    """Placeholder implementation so callers can wire retrieval without changing APIs."""

    async def index(
        self, session_id: UUID, content: str, metadata: dict[str, Any]
    ) -> None:
        return None

    async def search(
        self, session_id: UUID, query: str, limit: int = 5
    ) -> list[dict[str, Any]]:
        return []


def build_vector_retriever() -> VectorRetriever:
    return DisabledVectorRetriever()


__all__ = ["DisabledVectorRetriever", "VectorRetriever", "build_vector_retriever"]

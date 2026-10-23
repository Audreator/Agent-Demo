from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from app.citations.link_check import LinkStatus, normalize_url

MAX_TITLE = 200
MAX_SNIPPET = 300


def _clean_label(value: Any, limit: int) -> str:
    text = str(value or "").replace("\x00", "").strip()
    return text[:limit]


def merge_citations(
    groups: list[list[dict[str, Any]]],
    *,
    prefix: str = "",
    limit: int = 10,
) -> list[dict[str, Any]]:
    """同一批搜索结果按 URL 去重、保留首次出现顺序，最多 limit 条。"""
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for group in groups:
        for item in group:
            if not isinstance(item, dict):
                continue
            url = normalize_url(str(item.get("url") or ""))
            if not url or url in seen:
                continue
            seen.add(url)
            index = len(merged) + 1
            merged.append(
                {
                    "id": f"{prefix}c{index}" if prefix else f"c{index}",
                    "title": _clean_label(item.get("title"), MAX_TITLE),
                    "url": url,
                    "snippet": _clean_label(item.get("snippet"), MAX_SNIPPET),
                    "reachable": False,
                    "status_code": None,
                    "checked_at": None,
                    "reason": "",
                    "rejected": False,
                }
            )
            if len(merged) >= limit:
                return merged
    return merged


def apply_link_statuses(
    citations: list[dict[str, Any]], statuses: dict[str, LinkStatus]
) -> list[dict[str, Any]]:
    checked_at = datetime.now(UTC).isoformat()
    for item in citations:
        status = statuses.get(str(item.get("url") or ""))
        if status is None:
            continue
        item["reachable"] = status.reachable
        item["status_code"] = status.status_code
        item["reason"] = status.reason
        item["rejected"] = status.rejected
        item["checked_at"] = status.checked_at.isoformat() or checked_at
    return citations


def sanitize_citations(
    raw: Any, *, prefix: str = "", limit: int = 10
) -> list[dict[str, Any]]:
    """校验并裁剪来自工具结果的 citations，过滤非法链接。"""
    if not isinstance(raw, list):
        return []
    cleaned: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        url = normalize_url(str(entry.get("url") or ""))
        if not url or url in seen:
            continue
        seen.add(url)
        citation_id = str(entry.get("id") or "")
        if not citation_id:
            citation_id = f"{prefix}c{len(cleaned) + 1}" if prefix else f"c{len(cleaned) + 1}"
        cleaned.append(
            {
                "id": citation_id[:80],
                "title": _clean_label(entry.get("title"), MAX_TITLE),
                "url": url,
                "snippet": _clean_label(entry.get("snippet"), MAX_SNIPPET),
                "reachable": bool(entry.get("reachable")),
                "status_code": (
                    int(entry["status_code"])
                    if isinstance(entry.get("status_code"), int)
                    else None
                ),
                "checked_at": (
                    str(entry.get("checked_at"))
                    if entry.get("checked_at") is not None
                    else None
                ),
                "reason": _clean_label(entry.get("reason"), 200),
                "rejected": bool(entry.get("rejected")),
            }
        )
        if len(cleaned) >= limit:
            break
    return cleaned


def public_citation(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(item.get("id") or ""),
        "title": str(item.get("title") or "")[:MAX_TITLE],
        "url": str(item.get("url") or ""),
        "snippet": str(item.get("snippet") or "")[:MAX_SNIPPET],
        "reachable": bool(item.get("reachable")),
        "status_code": item.get("status_code"),
        "checked_at": item.get("checked_at"),
        "reason": str(item.get("reason") or ""),
    }


__all__ = [
    "apply_link_statuses",
    "merge_citations",
    "public_citation",
    "sanitize_citations",
]

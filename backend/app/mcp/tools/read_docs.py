from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from app.mcp.tools.artifacts import ArtifactError, read_artifact


class DocsError(ValueError):
    pass


_REF_PATTERN = re.compile(r"^artifact://([A-Za-z0-9_-]{1,128})$")


def docs_root() -> Path:
    root = Path(os.environ.get("DOCS_ROOT", "docs"))
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def _resolve_within_root(raw_path: str) -> Path:
    if not raw_path or "\x00" in raw_path:
        raise DocsError("path 不能为空")
    candidate = Path(raw_path)
    root = docs_root()
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise DocsError("非法路径：访问范围限制在 DOCS_ROOT 内") from exc
    if not resolved.is_file():
        raise DocsError(f"文件不存在: {raw_path}")
    return resolved


def read_document(
    *,
    path: str | None = None,
    ref: str | None = None,
    offset: int = 0,
    limit: int = 4000,
    session_id: str | None = None,
) -> dict[str, Any]:
    if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
        raise DocsError("offset 必须是 >= 0 的整数")
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1 or limit > 100_000:
        raise DocsError("limit 必须是 1-100000 的整数")

    if ref:
        matched = _REF_PATTERN.fullmatch(ref.strip())
        if not matched:
            raise DocsError("ref 必须是 artifact://{id} 格式")
        try:
            content = read_artifact(ref.strip(), session_id=session_id)
        except ArtifactError as exc:
            raise DocsError(str(exc)) from exc
        source = ref.strip()
    elif path:
        resolved = _resolve_within_root(path)
        content = resolved.read_text(encoding="utf-8", errors="replace")
        source = str(resolved)
    else:
        raise DocsError("path 与 ref 至少提供一个")

    total = len(content)
    if offset >= total and total > 0:
        chunk = ""
    else:
        chunk = content[offset : offset + limit]
    next_offset = offset + len(chunk)
    return {
        "source": source,
        "offset": offset,
        "limit": limit,
        "total_chars": total,
        "returned_chars": len(chunk),
        "truncated": next_offset < total,
        "next_offset": next_offset if next_offset < total else None,
        "content": chunk,
    }


__all__ = ["DocsError", "docs_root", "read_document"]

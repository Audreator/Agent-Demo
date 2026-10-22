from __future__ import annotations

import json
import os
import re
import uuid
from pathlib import Path

_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


class ArtifactError(ValueError):
    pass


def artifacts_dir() -> Path:
    path = Path(os.environ.get("ARTIFACTS_DIR", "artifacts"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def validate_artifact_id(artifact_id: str) -> str:
    if not _ID_PATTERN.fullmatch(artifact_id):
        raise ArtifactError("非法的 artifact id")
    return artifact_id


def artifact_path(artifact_id: str, base_dir: Path | None = None) -> Path:
    root = base_dir or artifacts_dir()
    return root / f"{validate_artifact_id(artifact_id)}.txt"


def _meta_path(artifact_id: str, base_dir: Path | None = None) -> Path:
    root = base_dir or artifacts_dir()
    return root / f"{validate_artifact_id(artifact_id)}.meta.json"


def save_artifact(
    content: str,
    artifact_id: str | None = None,
    *,
    session_id: str | None = None,
    tool_call_id: str | None = None,
    base_dir: Path | None = None,
) -> str:
    identifier = validate_artifact_id(artifact_id) if artifact_id else uuid.uuid4().hex
    root = base_dir or artifacts_dir()
    root.mkdir(parents=True, exist_ok=True)
    artifact_path(identifier, root).write_text(content, encoding="utf-8")
    _meta_path(identifier, root).write_text(
        json.dumps(
            {"session_id": session_id, "tool_call_id": tool_call_id},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return f"artifact://{identifier}"


def read_artifact(
    ref: str, *, session_id: str | None = None, base_dir: Path | None = None
) -> str:
    if not ref.startswith("artifact://"):
        raise ArtifactError("ref 必须以 artifact:// 开头")
    artifact_id = ref[len("artifact://") :]
    path = artifact_path(artifact_id, base_dir)
    meta_path = _meta_path(artifact_id, base_dir)
    if not path.is_file():
        raise ArtifactError(f"artifact 不存在: {ref}")
    if not meta_path.is_file():
        raise ArtifactError("artifact 缺少归属信息，拒绝读取")
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError("artifact 元数据损坏，拒绝读取") from exc
    owner = meta.get("session_id")
    if not owner:
        raise ArtifactError("artifact 缺少会话归属信息，拒绝读取")
    if not session_id or str(owner) != str(session_id):
        raise ArtifactError("artifact 不属于当前会话，拒绝读取")
    return path.read_text(encoding="utf-8", errors="replace")


__all__ = [
    "ArtifactError",
    "artifact_path",
    "artifacts_dir",
    "read_artifact",
    "save_artifact",
    "validate_artifact_id",
]

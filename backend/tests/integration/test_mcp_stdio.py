from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest
from sqlalchemy import func, select

from app.agent.tool_executor import ToolExecutor
from app.config import BASE_DIR, Settings
from app.errors import MCPProcessError, MCPStartError, MCPTimeoutError
from app.mcp.client import MCPClient
from app.models.db_models import Todo
from app.observability.tracing import get_tracer
from tests.helpers import FakeMCP


async def test_mcp_stdio_end_to_end(settings: Settings, db: Any) -> None:
    docs_root = Path(settings.docs_root)
    docs_root.mkdir(parents=True, exist_ok=True)
    (docs_root / "guide.md").write_text("第一行\n第二行 内容", encoding="utf-8")
    (docs_root / "big.md").write_text("B" * 90_000, encoding="utf-8")
    artifacts = Path(settings.artifacts_dir)
    artifacts.mkdir(parents=True, exist_ok=True)
    artifact_session = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    (artifacts / "demo-artifact.txt").write_text("artifact 内容", encoding="utf-8")
    (artifacts / "demo-artifact.meta.json").write_text(
        json.dumps({"session_id": artifact_session, "tool_call_id": "demo-artifact"}),
        encoding="utf-8",
    )

    client = MCPClient(
        settings.mcp_args,
        env=settings.mcp_env(),
        cwd=BASE_DIR,
        timeout=30,
    )
    await client.start()
    try:
        tools = await client.list_tools()
        assert {tool["name"] for tool in tools} == {
            "calculator",
            "todo",
            "search",
            "weather",
            "read_docs",
        }

        calc = await client.call_tool("calculator", {"expression": "0.1+0.2"})
        assert calc["isError"] is False
        calc_payload = json.loads(calc["content"][0]["text"])
        assert calc_payload["result"] == "0.3"

        args = {"action": "create", "content": "写周报", "client_op_id": "mcp-op-1"}
        first = await client.call_tool(
            "todo", args, user_id=settings.dev_user_id, session_id=None
        )
        assert first["isError"] is False
        second = await client.call_tool(
            "todo", args, user_id=settings.dev_user_id, session_id=None
        )
        payload = json.loads(second["content"][0]["text"])
        assert payload["idempotent_replay"] is True
        todo_id = payload["todo"]["id"]

        update_args = {
            "action": "update",
            "id": todo_id,
            "content": "写周报（更新）",
            "client_op_id": "mcp-op-update",
        }
        update_first = await client.call_tool(
            "todo", update_args, user_id=settings.dev_user_id
        )
        update_second = await client.call_tool(
            "todo", update_args, user_id=settings.dev_user_id
        )
        assert update_first["isError"] is False
        updated = json.loads(update_second["content"][0]["text"])
        assert updated["idempotent_replay"] is True
        assert updated["todo"]["content"] == "写周报（更新）"

        complete_args = {
            "action": "complete",
            "id": todo_id,
            "client_op_id": "mcp-op-complete",
        }
        complete_first = await client.call_tool(
            "todo", complete_args, user_id=settings.dev_user_id
        )
        complete_second = await client.call_tool(
            "todo", complete_args, user_id=settings.dev_user_id
        )
        assert complete_first["isError"] is False
        completed = json.loads(complete_second["content"][0]["text"])
        assert completed["idempotent_replay"] is True
        assert completed["todo"]["status"] == "done"

        create_again = await client.call_tool(
            "todo", args, user_id=settings.dev_user_id
        )
        again_payload = json.loads(create_again["content"][0]["text"])
        assert again_payload["todo"]["id"] == todo_id
        assert again_payload["idempotent_replay"] is True

        delete_args = {
            "action": "delete",
            "id": todo_id,
            "client_op_id": "mcp-op-delete",
        }
        delete_first = await client.call_tool(
            "todo", delete_args, user_id=settings.dev_user_id
        )
        delete_second = await client.call_tool(
            "todo", delete_args, user_id=settings.dev_user_id
        )
        assert delete_first["isError"] is False
        deleted = json.loads(delete_second["content"][0]["text"])
        assert deleted["deleted"] is True
        assert deleted["idempotent_replay"] is True

        listing = await client.call_tool("todo", {"action": "list"}, user_id=settings.dev_user_id)
        listing_payload = json.loads(listing["content"][0]["text"])
        assert listing_payload["count"] == 0

        count = (
            await db.execute(
                select(func.count())
                .select_from(Todo)
                .where(Todo.client_op_id == "mcp-op-1")
            )
        ).scalar_one()
        assert count == 1

        docs = await client.call_tool(
            "read_docs", {"path": "guide.md", "offset": 0, "limit": 4}
        )
        assert docs["isError"] is False
        docs_payload = json.loads(docs["content"][0]["text"])
        assert "第一行" in docs_payload["content"]
        assert docs_payload["truncated"] is True

        big_docs = await client.call_tool(
            "read_docs", {"path": "big.md", "offset": 0, "limit": 90_000}
        )
        assert big_docs["isError"] is False
        big_payload = json.loads(big_docs["content"][0]["text"])
        assert big_payload["returned_chars"] == 90_000
        assert big_payload["truncated"] is False
        (docs_root / "big.md").unlink(missing_ok=True)

        traversal = await client.call_tool("read_docs", {"path": "../conftest.py"})
        assert traversal["isError"] is True
        assert "DOCS_ROOT" in traversal["content"][0]["text"]

        artifact = await client.call_tool(
            "read_docs",
            {"ref": "artifact://demo-artifact"},
            session_id=artifact_session,
        )
        assert artifact["isError"] is False
        artifact_payload = json.loads(artifact["content"][0]["text"])
        assert artifact_payload["content"] == "artifact 内容"

        cross_session = await client.call_tool(
            "read_docs",
            {"ref": "artifact://demo-artifact"},
            session_id="bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
        )
        assert cross_session["isError"] is True
        assert "不属于当前会话" in cross_session["content"][0]["text"]
    finally:
        await client.close()


async def test_mcp_client_handles_process_start_failure(settings: Settings) -> None:
    client = MCPClient(["definitely-not-a-real-binary-xyz"], env={}, timeout=5)
    with pytest.raises(MCPStartError):
        await client.start()
    await client.close()


FAKE_SERVER = """
import json, sys
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    msg = json.loads(line)
    if "id" not in msg:
        continue
    method = msg.get("method")
    if method == "initialize":
        result = {"protocolVersion": "2024-11-05", "capabilities": {},
                  "serverInfo": {"name": "fake", "version": "0"}}
    elif method == "tools/list":
        result = {"tools": []}
    else:
        result = {}
    sys.stdout.write("this is a non-protocol log line\\n")
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": msg["id"], "result": result}) + "\\n")
    sys.stdout.flush()
"""

TIMEOUT_SERVER = """
import time
time.sleep(30)
"""

EXIT_SERVER = "import sys; sys.exit(0)"


async def test_mcp_client_ignores_non_json_lines() -> None:
    client = MCPClient([sys.executable, "-c", FAKE_SERVER], timeout=5)
    await client.start()
    try:
        assert await client.list_tools() == []
    finally:
        await client.close()


async def test_mcp_client_timeout() -> None:
    client = MCPClient([sys.executable, "-c", TIMEOUT_SERVER], timeout=0.5)
    with pytest.raises(MCPTimeoutError):
        await client.start()
    await client.close()


async def test_mcp_client_process_exit() -> None:
    client = MCPClient([sys.executable, "-c", EXIT_SERVER], timeout=3)
    with pytest.raises(MCPProcessError):
        await client.start()
    await client.close()


class FlakyMCP(FakeMCP):
    def __init__(self) -> None:
        super().__init__()
        self.attempts = 0

    async def call_tool(
        self, name: str, arguments: dict[str, Any], **kwargs: Any
    ) -> dict[str, Any]:
        self.attempts += 1
        if self.attempts < 3:
            return {
                "content": [{"type": "text", "text": "ToolFailure: transient"}],
                "isError": True,
            }
        return await super().call_tool(name, arguments, **kwargs)


async def test_tool_executor_retries_three_times(settings: Settings) -> None:
    retry_settings = settings.model_copy(update={"tool_retry": 3})
    flaky = FlakyMCP()
    executor = ToolExecutor(
        flaky,  # type: ignore[arg-type]
        retry_settings,
        tracer=get_tracer(retry_settings),
        user_id=retry_settings.dev_user_id,
    )
    result = await executor.execute(
        "calculator",
        {"expression": "1+1"},
        session_id=UUID("00000000-0000-0000-0000-000000000001"),
        turn_no=1,
        tool_call_id="retry-tool-call",
    )
    assert result.ok is True
    assert result.attempts == 3
    assert flaky.attempts == 3

from __future__ import annotations

import io
import json

import pytest

from app.mcp.schema import SchemaValidationError, validate_arguments
from app.mcp.server import (
    GuardedStdout,
    dispatch,
    install_stdout_guard,
    serve,
    stdout_purity_check,
)


@pytest.mark.asyncio
async def test_initialize_and_tools_list() -> None:
    init = await dispatch({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    assert init is not None
    assert init["result"]["protocolVersion"] == "2024-11-05"

    listed = await dispatch({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
    assert listed is not None
    tools = listed["result"]["tools"]
    assert {tool["name"] for tool in tools} == {
        "calculator",
        "todo",
        "search",
        "weather",
        "read_docs",
    }
    for tool in tools:
        assert tool["inputSchema"]["type"] == "object"
        assert tool["description"]


@pytest.mark.asyncio
async def test_tools_call_calculator() -> None:
    response = await dispatch(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "calculator", "arguments": {"expression": "0.1+0.2"}},
        }
    )
    assert response is not None
    assert response["result"]["isError"] is False
    payload = json.loads(response["result"]["content"][0]["text"])
    assert payload["result"] == "0.3"


@pytest.mark.asyncio
async def test_tools_call_invalid_schema() -> None:
    response = await dispatch(
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {"name": "calculator", "arguments": {"precision": 3}},
        }
    )
    assert response is not None
    assert response["error"]["code"] == -32602


@pytest.mark.asyncio
async def test_unknown_tool_and_method() -> None:
    unknown = await dispatch(
        {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {"name": "nope", "arguments": {}},
        }
    )
    assert unknown is not None
    assert unknown["error"]["code"] == -32602
    method = await dispatch({"jsonrpc": "2.0", "id": 6, "method": "nope", "params": {}})
    assert method is not None
    assert method["error"]["code"] == -32601


@pytest.mark.asyncio
async def test_notification_returns_no_response() -> None:
    response = await dispatch(
        {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}}
    )
    assert response is None


def test_schema_validation() -> None:
    schema = {
        "type": "object",
        "properties": {"days": {"type": "integer", "minimum": 1, "maximum": 7}},
        "required": ["days"],
        "additionalProperties": False,
    }
    validate_arguments(schema, {"days": 3})
    with pytest.raises(SchemaValidationError):
        validate_arguments(schema, {})
    with pytest.raises(SchemaValidationError):
        validate_arguments(schema, {"days": True})
    with pytest.raises(SchemaValidationError):
        validate_arguments(schema, {"days": 9})
    with pytest.raises(SchemaValidationError):
        validate_arguments(schema, {"days": 2, "extra": 1})


def test_stdout_guard_discards_non_protocol_lines(monkeypatch: pytest.MonkeyPatch) -> None:
    target = io.StringIO()
    stderr = io.StringIO()
    monkeypatch.setattr("sys.stderr", stderr)
    guard = GuardedStdout(target)
    guard.write("this is a log line\n")
    guard.write('{"hello": "world"}\n')
    guard.write('{"jsonrpc":"2.0","id":1,"result":{}}\n')
    guard.flush()
    assert "jsonrpc" in target.getvalue()
    assert "log line" not in target.getvalue()
    assert "hello" not in target.getvalue()
    assert "discarded non-protocol output" in stderr.getvalue()


def test_stdout_purity_check(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.stdout", io.StringIO())
    install_stdout_guard()
    check = stdout_purity_check()
    assert check["pure"] is True
    assert check["guarded"] is True


@pytest.mark.asyncio
async def test_serve_round_trip() -> None:
    stdin = io.StringIO(
        '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}\n'
        '{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}\n'
    )
    stdout = io.StringIO()
    await serve(stdin=stdin, stdout=stdout)
    lines = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert lines[0]["id"] == 1
    assert lines[1]["id"] == 2
    assert len(lines[1]["result"]["tools"]) == 5

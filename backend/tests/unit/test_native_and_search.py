from __future__ import annotations

import pytest

from app.agent.native_adapter import normalize_native, tools_as_openai
from app.models import search as search_module
from app.models.search import (
    DeepSeekWebSearch,
    SearchError,
    _extract_responses_results,
    enforce_size_limit,
)

REGISTRY = {
    "calculator": {
        "name": "calculator",
        "description": "calc",
        "inputSchema": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    }
}


def test_native_tool_call_with_reasoning() -> None:
    response = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "reasoning_content": "先计算",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "calculator",
                                "arguments": '{"expression": "1+2"}',
                            },
                        }
                    ],
                }
            }
        ]
    }
    parsed = normalize_native(response, REGISTRY)
    assert parsed.kind == "action"
    assert parsed.name == "calculator"
    assert parsed.arguments == {"expression": "1+2"}
    assert parsed.thought == "先计算"
    assert parsed.call_id == "call_1"


def test_native_final_answer() -> None:
    response = {"choices": [{"message": {"content": "答案", "reasoning_content": "思考"}}]}
    parsed = normalize_native(response, REGISTRY)
    assert parsed.kind == "final"
    assert parsed.answer == "答案"


def test_native_empty_is_invalid() -> None:
    parsed = normalize_native({"choices": [{"message": {"content": ""}}]}, REGISTRY)
    assert parsed.kind == "invalid"


def test_native_unknown_tool() -> None:
    response = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {"function": {"name": "shell", "arguments": "{}"}}
                    ]
                }
            }
        ]
    }
    assert normalize_native(response, REGISTRY).kind == "invalid"


def test_native_multiple_tool_calls_are_rejected() -> None:
    response = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {"name": "calculator", "arguments": '{"expression":"1"}'},
                        },
                        {
                            "id": "call_2",
                            "function": {"name": "calculator", "arguments": '{"expression":"2"}'},
                        },
                    ]
                }
            }
        ]
    }
    parsed = normalize_native(response, REGISTRY)
    assert parsed.kind == "invalid"
    assert "一次只能调用一个工具" in parsed.reason


def test_tools_as_openai() -> None:
    tools = tools_as_openai(REGISTRY)
    assert tools[0]["type"] == "function"
    assert tools[0]["function"]["name"] == "calculator"


def test_search_size_limit_marks_truncated() -> None:
    results = [
        {"title": f"t{i}", "url": f"https://example.com/{i}", "snippet": "x" * 400}
        for i in range(10)
    ]
    kept, truncated = enforce_size_limit(results, 1000)
    assert truncated is True
    assert len(kept) < len(results)
    assert all(len(item["snippet"]) <= 201 for item in kept)


def test_responses_api_web_search_results() -> None:
    data = {
        "output": [
            {"type": "web_search_call", "status": "completed"},
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": "杭州今天晴，气温 18-26 度。",
                        "annotations": [
                            {
                                "type": "url_citation",
                                "title": "杭州天气预报",
                                "url": "https://weather.example.com/hz",
                                "start_index": 0,
                                "end_index": 8,
                            }
                        ],
                    }
                ],
            },
        ]
    }
    results = _extract_responses_results(data)
    assert len(results) == 1
    assert results[0]["url"] == "https://weather.example.com/hz"
    assert results[0]["title"] == "杭州天气预报"
    assert results[0]["snippet"]


class FakeSearchResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.status_code = 200
        self.text = ""

    def json(self) -> dict:
        return self.payload


class FakeSearchClient:
    captured: dict = {}

    def __init__(self, *args, **kwargs) -> None:
        pass

    async def __aenter__(self) -> FakeSearchClient:
        return self

    async def __aexit__(self, *args) -> None:
        return None

    async def post(self, url: str, json: dict, headers: dict) -> FakeSearchResponse:
        FakeSearchClient.captured = {"url": url, "json": json, "headers": headers}
        return FakeSearchResponse(
            {
                "output": [
                    {"type": "web_search_call", "status": "completed"},
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "杭州天气晴。",
                                "annotations": [
                                    {
                                        "type": "url_citation",
                                        "title": "天气",
                                        "url": "https://weather.example.com/a",
                                        "start_index": 0,
                                        "end_index": 4,
                                    }
                                ],
                            }
                        ],
                    },
                ]
            }
        )


async def test_responses_search_request_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(search_module.httpx, "AsyncClient", FakeSearchClient)
    provider = DeepSeekWebSearch(
        base_url="https://api.deepseek.com",
        api_key="test-key",
        model="deepseek-v4-pro",
    )
    results = await provider.search("杭州天气", max_results=3)
    captured = FakeSearchClient.captured
    assert captured["url"] == "https://api.deepseek.com/responses"
    assert captured["json"]["model"] == "deepseek-v4-pro"
    assert captured["json"]["tools"] == [{"type": "web_search"}]
    assert captured["headers"]["Authorization"] == "Bearer test-key"
    assert results[0]["url"] == "https://weather.example.com/a"


class FakeNoCitationClient(FakeSearchClient):
    async def post(self, url: str, json: dict, headers: dict) -> FakeSearchResponse:
        return FakeSearchResponse(
            {
                "output": [
                    {"type": "web_search_call", "status": "completed"},
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "编造的结果",
                                "annotations": [],
                            }
                        ],
                    },
                ]
            }
        )


async def test_responses_search_without_citations_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(search_module.httpx, "AsyncClient", FakeNoCitationClient)
    provider = DeepSeekWebSearch(
        base_url="https://api.deepseek.com",
        api_key="test-key",
        model="deepseek-v4-pro",
    )
    with pytest.raises(SearchError, match="url_citation"):
        await provider.search("杭州天气", max_results=3)

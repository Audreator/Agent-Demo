from __future__ import annotations

import pytest

from app.mcp.tools import weather as weather_module
from app.mcp.tools.weather import WeatherError, get_weather


class FakeResponse:
    def __init__(self, payload: dict, status: int = 200) -> None:
        self.payload = payload
        self.status_code = status

    def json(self) -> dict:
        return self.payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"status={self.status_code}")


class FakeClient:
    def __init__(self, *args, **kwargs) -> None:
        pass

    async def __aenter__(self) -> FakeClient:
        return self

    async def __aexit__(self, *args) -> None:
        return None

    async def get(self, url: str, params: dict) -> FakeResponse:
        if "geocoding" in url:
            if params.get("name") == "不存在的地方":
                return FakeResponse({"results": []})
            return FakeResponse(
                {
                    "results": [
                        {
                            "name": "杭州",
                            "admin1": "浙江省",
                            "country": "中国",
                            "latitude": 30.29,
                            "longitude": 120.16,
                        }
                    ]
                }
            )
        return FakeResponse(
            {
                "timezone": "Asia/Shanghai",
                "daily": {
                    "time": ["2026-10-21", "2026-10-22", "2026-10-23"],
                    "weather_code": [0, 61, 3],
                    "temperature_2m_max": [26.0, 22.0, 24.0],
                    "temperature_2m_min": [16.0, 15.0, 17.0],
                    "precipitation_probability_max": [5, 80, 30],
                },
            }
        )


async def test_weather_normalizes_chinese_description(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(weather_module.httpx, "AsyncClient", FakeClient)
    result = await get_weather("杭州", days=3)
    assert result["location"] == "杭州浙江省中国"
    assert result["units"] == "metric"
    assert len(result["days"]) == 3
    assert result["days"][0]["weather"] == "晴"
    assert result["days"][1]["weather"] == "小雨"
    assert result["days"][0]["weekday"] in {"周一", "周二", "周三", "周四", "周五", "周六", "周日"}
    assert "℃" in result["summary"]
    assert "降水概率 80%" in result["summary"]


async def test_weather_invalid_days_and_location(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(weather_module.httpx, "AsyncClient", FakeClient)
    with pytest.raises(WeatherError):
        await get_weather("杭州", days=0)
    with pytest.raises(WeatherError):
        await get_weather("")
    with pytest.raises(WeatherError):
        await get_weather("不存在的地方")

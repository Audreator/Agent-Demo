from __future__ import annotations

import re
from datetime import datetime
from typing import Any

import httpx

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

WEATHER_CODES: dict[int, str] = {
    0: "晴",
    1: "大部晴朗",
    2: "局部多云",
    3: "阴",
    45: "雾",
    48: "雾凇",
    51: "小毛毛雨",
    53: "毛毛雨",
    55: "强毛毛雨",
    56: "冻毛毛雨",
    57: "强冻毛毛雨",
    61: "小雨",
    63: "中雨",
    65: "大雨",
    66: "冻雨",
    67: "强冻雨",
    71: "小雪",
    73: "中雪",
    75: "大雪",
    77: "雪粒",
    80: "小阵雨",
    81: "阵雨",
    82: "强阵雨",
    85: "小阵雪",
    86: "强阵雪",
    95: "雷阵雨",
    96: "雷阵雨伴小冰雹",
    99: "雷阵雨伴大冰雹",
}


class WeatherError(ValueError):
    pass


def _sanitize_location(location: str) -> str:
    value = location.strip()
    value = re.sub(r"[\x00-\x1f]", "", value)
    return value[:80]


async def _geocode(client: httpx.AsyncClient, location: str) -> dict[str, Any]:
    response = await client.get(
        GEOCODE_URL,
        params={"name": location, "count": 1, "language": "zh", "format": "json"},
    )
    response.raise_for_status()
    data = response.json()
    results = data.get("results") or []
    if not results:
        raise WeatherError(f"无法解析地名：{location}，请补充更具体的城市名")
    return results[0]


async def get_weather(location: str, days: int = 1, units: str = "metric") -> dict[str, Any]:
    location = _sanitize_location(location or "")
    if not location:
        raise WeatherError("location 不能为空")
    if not isinstance(days, int) or isinstance(days, bool) or not 1 <= days <= 7:
        raise WeatherError("days 必须是 1-7 的整数")
    if units not in {"metric", "imperial"}:
        raise WeatherError("units 只能是 metric 或 imperial")

    temperature_unit = "celsius" if units == "metric" else "fahrenheit"
    wind_unit = "kmh" if units == "metric" else "mph"
    temp_symbol = "℃" if units == "metric" else "℉"

    async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
        place = await _geocode(client, location)
        params: dict[str, Any] = {
            "latitude": place["latitude"],
            "longitude": place["longitude"],
            "daily": "weather_code,temperature_2m_max,temperature_2m_min,"
            "precipitation_probability_max",
            "timezone": "auto",
            "forecast_days": days,
            "temperature_unit": temperature_unit,
            "wind_speed_unit": wind_unit,
        }
        response = await client.get(FORECAST_URL, params=params)
        response.raise_for_status()
        forecast = response.json()

    daily = forecast.get("daily") or {}
    dates = daily.get("time") or []
    if not dates:
        raise WeatherError("天气服务未返回预报数据")

    result_days: list[dict[str, Any]] = []
    for index, date in enumerate(dates[:days]):
        code = _at(daily.get("weather_code"), index)
        max_temp = _at(daily.get("temperature_2m_max"), index)
        min_temp = _at(daily.get("temperature_2m_min"), index)
        rain = _at(daily.get("precipitation_probability_max"), index)
        description = WEATHER_CODES.get(int(code), "未知天气") if code is not None else "未知天气"
        result_days.append(
            {
                "date": date,
                "weekday": _weekday(date),
                "weather": description,
                "weather_code": code,
                "temp_max": max_temp,
                "temp_min": min_temp,
                "precipitation_probability": rain,
            }
        )

    place_name = "".join(
        part
        for part in (
            place.get("name"),
            place.get("admin1"),
            place.get("country"),
        )
        if part
    )
    text_lines = []
    for item in result_days:
        text_lines.append(
            f"{item['date']} {item['weekday']} {item['weather']} "
            f"{item['temp_min']}{temp_symbol}~{item['temp_max']}{temp_symbol}"
            + (
                f"，降水概率 {item['precipitation_probability']}%"
                if item["precipitation_probability"] is not None
                else ""
            )
        )
    return {
        "location": place_name or location,
        "latitude": place.get("latitude"),
        "longitude": place.get("longitude"),
        "timezone": forecast.get("timezone"),
        "units": units,
        "days": result_days,
        "summary": "；".join(text_lines),
    }


def _at(values: Any, index: int) -> Any:
    if isinstance(values, list) and index < len(values):
        return values[index]
    return None


def _weekday(date_text: str) -> str:
    try:
        value = datetime.strptime(date_text, "%Y-%m-%d")
    except ValueError:
        return ""
    names = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    return names[value.weekday()]


__all__ = ["WeatherError", "get_weather"]

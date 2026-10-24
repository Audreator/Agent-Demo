from __future__ import annotations

import re

REFUSAL_PATTERNS = (
    "我不知道",
    "不知道",
    "我不清楚",
    "不清楚",
    "我不了解",
    "不了解",
    "我不确定",
    "我没找到",
    "没找到",
    "我无法回答",
    "无法回答",
    "无法确定",
    "无法确认",
    "无法得知",
    "无法提供",
    "无法给出",
    "没有相关信息",
    "暂无相关信息",
    "未找到相关",
    "未能找到",
    "找不到相关",
    "没有找到相关",
    "没有查到",
    "没有足够的信息",
    "i don't know",
    "i do not know",
    "i'm not sure",
    "i am not sure",
    "no information",
    "cannot answer",
    "can't answer",
    "unable to answer",
)

REFUSAL_RE = re.compile("|".join(re.escape(item) for item in REFUSAL_PATTERNS), re.IGNORECASE)
URL_RE = re.compile(r"https?://", re.IGNORECASE)

FORCED_SEARCH_EMPTY_TEXT = "已尝试联网检索，暂未获取到相关信息。"

FINAL_KINDS = {"final", "final_answer"}


def is_refusal_answer(
    answer: str, *, match_window: int = 200, max_answer_len: int = 800
) -> bool:
    text = (answer or "").strip().replace("’", "'")
    if not text or len(text) > max_answer_len:
        return False
    if URL_RE.search(text):
        return False
    match = REFUSAL_RE.search(text)
    if match is None:
        return False
    return match.start() < match_window


def forced_search_query(user_input: str, max_chars: int) -> str:
    """Normalize the user input used for the forced search request."""
    return (user_input or "").strip()[: max(0, max_chars)]


def should_force_search(
    *,
    answer: str,
    has_ok_search: bool,
    already_forced: bool,
    enabled: bool,
    finish_kind: str,
    match_window: int = 200,
    max_answer_len: int = 800,
) -> bool:
    if not enabled or already_forced or has_ok_search:
        return False
    if finish_kind not in FINAL_KINDS:
        return False
    return is_refusal_answer(
        answer, match_window=match_window, max_answer_len=max_answer_len
    )


__all__ = [
    "FORCED_SEARCH_EMPTY_TEXT",
    "REFUSAL_PATTERNS",
    "forced_search_query",
    "is_refusal_answer",
    "should_force_search",
]

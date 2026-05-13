"""
app/services/progress_tracker.py
Derives real-time step labels and progress percentages from graph state chunks.
Also contains MessageBuffer and agent status helpers.
"""
from __future__ import annotations

import datetime
import math
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

from app.services.session_serialization import SECTION_TITLES, DEFAULT_AGENT_NAMES


# ── Analyst metadata ────────────────────────────────────────────────────────

ANALYST_ORDER = ["market", "social", "news", "fundamentals", "alpha"]

ANALYST_DISPLAY: Dict[str, str] = {
    "market":       "Market Analyst",
    "social":       "Social Analyst",
    "news":         "News Analyst",
    "fundamentals": "Fundamentals Analyst",
    "alpha":        "Alpha Analyst",
}

REPORT_KEY: Dict[str, str] = {
    "market":       "market_report",
    "social":       "sentiment_report",
    "news":         "news_report",
    "fundamentals": "fundamentals_report",
    "alpha":        "quant_report",
}

TOOL_ANALYST: Dict[str, str] = {
    "get_stock_data":           "market",
    "get_indicators":           "market",
    "get_market_context":       "market",
    "get_f247_forum_posts":     "social",
    "get_ticker_news":          "social",
    "get_news":                 "news",
    "get_global_news":          "news",
    "get_insider_transactions": "news",
    "get_fundamentals":         "fundamentals",
    "get_balance_sheet":        "fundamentals",
    "get_cashflow":             "fundamentals",
    "get_income_statement":     "fundamentals",
}




# ── Chunk helpers ────────────────────────────────────────────────────────────

def _get_last_message(chunk: Dict[str, Any]) -> Any:
    msgs = chunk.get("messages")
    if not isinstance(msgs, list) or not msgs:
        return None
    return msgs[-1]


def last_msg_tool_calls(chunk: Dict[str, Any]) -> List[str]:
    msg = _get_last_message(chunk)
    if msg is None:
        return []
    raw = (
        msg.get("tool_calls") if isinstance(msg, dict)
        else getattr(msg, "tool_calls", None)
    )
    if not isinstance(raw, list):
        return []
    names = []
    for call in raw:
        name = (
            call.get("name") or call.get("tool") if isinstance(call, dict)
            else getattr(call, "name", None)
        )
        if isinstance(name, str) and name:
            names.append(name)
    return names


def last_msg_is_clear(chunk: Dict[str, Any]) -> bool:
    msg = _get_last_message(chunk)
    if msg is None:
        return False
    if isinstance(msg, dict):
        role    = msg.get("role", "") or msg.get("type", "")
        content = msg.get("content", "")
    else:
        role    = getattr(msg, "type", "") or getattr(msg, "role", "")
        content = getattr(msg, "content", "")
    return (
        str(role).lower() in ("human", "user", "humanmessage")
        and str(content).strip() == "Continue"
    )


def to_positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
        return parsed if parsed > 0 else default
    except (TypeError, ValueError):
        return default





# ── MessageBuffer ────────────────────────────────────────────────────────────

from app.services.session_serialization import _normalize_section


class MessageBuffer:
    def __init__(self, max_length: int = 100):
        self.messages:        deque = deque(maxlen=max_length)
        self.tool_calls:      deque = deque(maxlen=max_length)
        self.current_report:  Optional[str] = None
        self.final_report:    Optional[str] = None
        self.agent_status:    Dict[str, str] = {a: "pending" for a in DEFAULT_AGENT_NAMES}
        self.current_agent:   Optional[str] = None
        self.report_sections: Dict[str, Optional[str]] = {k: None for k in SECTION_TITLES}

    def add_message(self, message_type: str, content: str) -> None:
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self.messages.append({"timestamp": ts, "type": message_type, "content": content})

    def add_tool_call(self, tool_name: str, args: Any) -> None:
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self.tool_calls.append({"timestamp": ts, "tool": tool_name, "args": args})

    def update_agent_status(self, agent: str, status: str) -> None:
        if agent not in self.agent_status:
            return
        if self.agent_status[agent] == "completed" and status != "completed":
            return
        self.agent_status[agent] = status
        if status == "in_progress":
            self.current_agent = agent

    def update_report_section(self, section_name: str, content: str) -> None:
        if section_name not in self.report_sections:
            return
        self.report_sections[section_name] = content
        title = SECTION_TITLES.get(section_name, section_name)
        self.current_report = f"### {title}\n{content}"
        self._rebuild_final_report()

    def _rebuild_final_report(self) -> None:
        parts: List[str] = []
        analyst_secs = [
            "market_report", "sentiment_report", "news_report",
            "fundamentals_report", "quant_report",
        ]
        if any(self.report_sections[s] for s in analyst_secs):
            parts.append("## Báo cáo nhóm phân tích")
            for sec in analyst_secs:
                val = self.report_sections.get(sec)
                if val:
                    parts.append(_normalize_section(SECTION_TITLES[sec], str(val)))
        for key, label in [
            ("investment_plan",        "## ===== QUYẾT ĐỊNH NHÓM NGHIÊN CỨU ====="),
            ("trader_investment_plan", "## ===== KẾ HOẠCH NHÓM GIAO DỊCH ====="),
            ("final_trade_decision",   "## ===== QUYẾT ĐỊNH CUỐI CÙNG ====="),
        ]:
            if self.report_sections.get(key):
                parts += ["---", label, str(self.report_sections[key])]
        self.final_report = "\n\n".join(parts) if parts else None
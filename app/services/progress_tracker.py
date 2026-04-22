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

ANALYST_ORDER = ["market", "social", "news", "fundamentals"]

ANALYST_DISPLAY: Dict[str, str] = {
    "market":       "Market Analyst",
    "social":       "Social Analyst",
    "news":         "News Analyst",
    "fundamentals": "Fundamentals Analyst",
}

REPORT_KEY: Dict[str, str] = {
    "market":       "market_report",
    "social":       "sentiment_report",
    "news":         "news_report",
    "fundamentals": "fundamentals_report",
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

TOOL_STEP_LABEL: Dict[str, str] = {
    "get_stock_data":           "Đang crawl dữ liệu giá",
    "get_indicators":           "Đang tính technical indicators",
    "get_market_context":       "Đang lấy bối cảnh thị trường",
    "get_f247_forum_posts":     "Đang thu thập thảo luận diễn đàn F247",
    "get_ticker_news":          "Đang crawl tin tức theo mã",
    "get_news":                 "Đang thu thập tin tức doanh nghiệp",
    "get_global_news":          "Đang tổng hợp tin tức vĩ mô",
    "get_insider_transactions": "Đang lấy giao dịch nội bộ",
    "get_fundamentals":         "Đang lấy dữ liệu fundamentals",
    "get_balance_sheet":        "Đang lấy bảng cân đối kế toán",
    "get_cashflow":             "Đang lấy báo cáo lưu chuyển tiền tệ",
    "get_income_statement":     "Đang lấy báo cáo kết quả kinh doanh",
}

ANALYST_RUNNING_LABEL: Dict[str, str] = {
    "market":       "Market Analyst đang phân tích...",
    "social":       "Social Analyst đang phân tích...",
    "news":         "News Analyst đang phân tích...",
    "fundamentals": "Fundamentals Analyst đang phân tích...",
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


# ── Progress derivation ──────────────────────────────────────────────────────

def derive_realtime_step(
    chunk: Dict[str, Any],
    max_debate_rounds: int = 1,
    max_risk_rounds: int = 1,
    selected_analysts: Optional[List[str]] = None,
) -> Optional[Tuple[str, int]]:
    max_debate_rounds = to_positive_int(max_debate_rounds, 1)
    max_risk_rounds   = to_positive_int(max_risk_rounds,   1)

    active = [k for k in ANALYST_ORDER if k in (selected_analysts or ANALYST_ORDER)]
    n = max(1, len(active))
    A_START, A_END = 15, 72

    done_keys = [k for k in active if chunk.get(REPORT_KEY[k])]
    n_done    = len(done_keys)

    # A. Tool call in progress
    tool_names = last_msg_tool_calls(chunk)
    if tool_names:
        first_tool  = tool_names[0]
        analyst_key = TOOL_ANALYST.get(first_tool)
        label = TOOL_STEP_LABEL.get(first_tool, "Đang thu thập dữ liệu...")
        slot_w = (A_END - A_START) / n
        if analyst_key and analyst_key in active:
            slot = active.index(analyst_key)
            pct  = int(A_START + slot * slot_w + slot_w / 2)
        else:
            pct = int(A_START + (A_END - A_START) * n_done / n)
        return (label, pct)

    # B. Final decision done
    if chunk.get("final_trade_decision"):
        return ("Hoàn thành phán quyết cuối cùng", 100)

    # C. All analysts done → downstream stages
    if n_done == n:
        slot_w = (A_END - A_START) / n

        if chunk.get("trader_investment_plan"):
            risk_state = chunk.get("risk_debate_state")
            if isinstance(risk_state, dict):
                ls = str(risk_state.get("latest_speaker", ""))
                jd = str(risk_state.get("judge_decision", ""))
                if jd or ls == "Judge":
                    return ("Portfolio Manager đang chốt quyết định cuối...", 99)
                max_t   = max(1, 3 * max_risk_rounds)
                count   = to_positive_int(risk_state.get("count"), 1)
                clamped = min(count, max_t)
                spct    = 90 + int((clamped / max_t) * 8)
                _rl = {
                    "Risky":   "Risky Analyst đang phân tích rủi ro...",
                    "Safe":    "Safe Analyst đang phân tích rủi ro...",
                    "Neutral": "Neutral Analyst đang phân tích rủi ro...",
                }
                if ls in _rl:
                    return (_rl[ls], spct)
            return ("Risky Analyst đang bắt đầu phân tích rủi ro...", 91)

        if chunk.get("investment_plan"):
            return ("Trader đang lập kế hoạch giao dịch...", 89)

        invest_state = chunk.get("investment_debate_state")
        if isinstance(invest_state, dict):
            jd    = str(invest_state.get("judge_decision", ""))
            count = to_positive_int(invest_state.get("count"), 0)
            if jd:
                return ("Research Manager đang chốt phán quyết...", 87)
            if count > 0:
                round_num = math.ceil(count / 2)
                round_num = min(round_num, max_debate_rounds)
                dpct = 78 + int((count / (2 * max_debate_rounds)) * 8)
                dpct = min(dpct, 86)
                is_bull = (count % 2 == 1)
                speaker = "Bull Researcher" if is_bull else "Bear Researcher"
                return (
                    f"{speaker} đang tranh luận (vòng {round_num}/{max_debate_rounds})...",
                    dpct,
                )

        if chunk.get("quant_report"):
            return ("Bull Researcher đang bắt đầu tranh luận...", 78)

        return ("AlphaGPT Analyst đang xử lý...", 74)

    # D. Analyst team in progress
    slot_w = (A_END - A_START) / n
    if n_done < n:
        current_key = next((k for k in active if not chunk.get(REPORT_KEY[k])), None)
        if current_key:
            slot = active.index(current_key)
            pct  = int(A_START + slot * slot_w + slot_w / 3)
            return (ANALYST_RUNNING_LABEL.get(current_key, f"{current_key} đang phân tích..."), pct)

    # E. Msg Clear transition
    if last_msg_is_clear(chunk):
        for key in active:
            if not chunk.get(REPORT_KEY[key]):
                slot = active.index(key)
                pct  = int(A_START + slot * slot_w)
                return (ANALYST_RUNNING_LABEL.get(key, f"{key} đang bắt đầu..."), pct)

    return None


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
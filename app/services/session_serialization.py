"""
app/services/session_serialization.py

Serialization helpers cho session data.
Đây là single source of truth cho:
- rebuild_reports_from_final_state()
- rebuild_agent_status_from_final_state()
- SECTION_TITLES, ORDERED_SECTIONS, DEFAULT_AGENT_NAMES (constants)

main.py import từ đây, không define lại.
"""
import datetime
import re
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants — dùng chung giữa main.py và session_serialization.py
# ---------------------------------------------------------------------------

SECTION_TITLES: Dict[str, str] = {
    "market_report":         "Phân tích thị trường",
    "sentiment_report":      "Phân tích tâm lý xã hội",
    "news_report":           "Phân tích tin tức",
    "fundamentals_report":   "Phân tích tài chính doanh nghiệp",
    "quant_report":          "Phân tích định lượng (AlphaGPT)",
    "investment_plan":       "Quyết định nhóm nghiên cứu",
    "trader_investment_plan":"Kế hoạch nhóm giao dịch",
    "final_trade_decision":  "Quyết định cuối cùng",
}

ORDERED_SECTIONS: List[str] = [
    "market_report",
    "sentiment_report",
    "news_report",
    "fundamentals_report",
    "quant_report",
    "investment_plan",
    "trader_investment_plan",
    "final_trade_decision",
]

# Tên các analyst agents theo đúng thứ tự hiển thị
DEFAULT_AGENT_NAMES: List[str] = [
    # Analyst team
    "Market Analyst",
    "Social Analyst",
    "News Analyst",
    "Fundamentals Analyst",
    "AlphaGPT Analyst",

    # Researcher team
    "Bull Researcher",
    "Bear Researcher",
    "Research Manager",

    # Trader
    "Trader",

    # Risk management
    "Risky Analyst",
    "Safe Analyst",
    "Neutral Analyst",
    "Portfolio Manager",
]

# Map section key → agent name (dùng để rebuild agent status từ final_state)
SECTION_TO_AGENT: Dict[str, str] = {
    "market_report":         "Market Analyst",
    "sentiment_report":      "Social Analyst",
    "news_report":           "News Analyst",
    "fundamentals_report":   "Fundamentals Analyst",
    "quant_report":          "AlphaGPT Analyst",
    "investment_plan":       "Research Manager",
    "trader_investment_plan":"Trader",
    "final_trade_decision":  "Portfolio Manager",
}


# ---------------------------------------------------------------------------
# Rebuild helpers — single source of truth
# ---------------------------------------------------------------------------

def _normalize_section(title: str, content: str) -> str:
    """
    Wrap một section analyst vào block chuẩn:
    - Đổi tên heading bên trong thành h3/h4/h5 (shift xuống 2 bậc)
      để không xung đột với h2 wrapper bên ngoài
    - Tách FINAL TRANSACTION PROPOSAL ra thành dòng riêng ở cuối
    """
    if not content:
        return ""

    # Shift tất cả heading bên trong xuống 2 bậc (# → ###, ## → ####, v.v.)
    def shift_heading(m):
        hashes = m.group(1)
        rest   = m.group(2)
        new_level = min(6, len(hashes) + 2)
        return '#' * new_level + rest

    normalized = re.sub(r'^(#{1,6})([ \t].+)$', shift_heading, content,
                        flags=re.MULTILINE)

    # Tách proposal ra khỏi body
    proposal_pattern = re.compile(
        r'\n?[^\n]*FINAL TRANSACTION PROPOSAL[^\n]*\n?',
        re.IGNORECASE
    )
    proposals = proposal_pattern.findall(normalized)
    body      = proposal_pattern.sub('', normalized).strip()

    result = f"## {title}\n\n{body}"
    if proposals:
        # Lấy decision value từ proposal
        m = re.search(r'(BUY|SELL|HOLD)', proposals[-1], re.IGNORECASE)
        decision_val = m.group(1).upper() if m else proposals[-1].strip()
        result += f"\n\n> **Khuyến nghị:** {decision_val}"

    return result

def rebuild_reports_from_final_state(
    final_state: Any,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Từ final_state dict, tái tạo:
    - current_report: section cuối cùng có nội dung
    - final_report:   toàn bộ báo cáo ghép lại

    Dùng khi load session từ disk (không có MessageBuffer).
    """
    if not isinstance(final_state, dict):
        return None, None

    # Tìm section cuối cùng có nội dung
    latest_key = latest_content = None
    for sec in ORDERED_SECTIONS:
        val = final_state.get(sec)
        if val:
            latest_key = sec
            latest_content = str(val)

    current_report: Optional[str] = None
    if latest_key and latest_content:
        title = SECTION_TITLES.get(latest_key, latest_key)
        current_report = f"### {title}\n{latest_content}"

    # Ghép full report
    analyst_sections = [
        "market_report", "sentiment_report",
        "news_report", "fundamentals_report", "quant_report",
    ]
    report_parts: List[str] = []

    if any(final_state.get(s) for s in analyst_sections):
        report_parts.append("## Báo cáo nhóm phân tích")
        for sec in analyst_sections:
            val = final_state.get(sec)
            if val:
                report_parts.append(_normalize_section(SECTION_TITLES[sec], str(val)))

    if final_state.get("investment_plan"):
        report_parts += [
            "---",
            "## ===== QUYẾT ĐỊNH NHÓM NGHIÊN CỨU =====",
            str(final_state["investment_plan"]),
        ]
    if final_state.get("trader_investment_plan"):
        report_parts += [
            "---",
            "## ===== KẾ HOẠCH NHÓM GIAO DỊCH =====",
            str(final_state["trader_investment_plan"]),
        ]
    if final_state.get("final_trade_decision"):
        report_parts += [
            "---",
            "## ===== QUYẾT ĐỊNH CUỐI CÙNG =====",
            str(final_state["final_trade_decision"]),
        ]

    final_report = "\n\n".join(report_parts) if report_parts else None
    return current_report, final_report


def rebuild_agent_status_from_final_state(
    final_state: Any,
    session_status: str,
) -> Dict[str, str]:
    """
    Tái tạo agent status map từ final_state.
    Dùng khi load session từ disk.
    """
    default_status = {a: "pending" for a in DEFAULT_AGENT_NAMES}

    if session_status == "completed":
        return {a: "completed" for a in default_status}

    if not isinstance(final_state, dict):
        return default_status

    status_map = dict(default_status)
    for sec, agent in SECTION_TO_AGENT.items():
        if final_state.get(sec):
            status_map[agent] = "completed"

    # Nếu investment_plan có → Bull/Bear cũng xong
    if final_state.get("investment_plan"):
        status_map["Bull Researcher"] = "completed"
        status_map["Bear Researcher"] = "completed"

    # Nếu final_trade_decision có → Risky/Safe/Neutral cũng xong
    if final_state.get("final_trade_decision"):
        for a in ["Risky Analyst", "Safe Analyst", "Neutral Analyst"]:
            status_map[a] = "completed"

    return status_map


# ---------------------------------------------------------------------------
# Serialization utilities
# ---------------------------------------------------------------------------

def to_jsonable(value: Any, depth: int = 0) -> Any:
    if depth > 6:
        return str(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (datetime.date, datetime.datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v, depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, deque)):
        return [to_jsonable(v, depth + 1) for v in value]
    if hasattr(value, "model_dump"):
        try:
            return to_jsonable(value.model_dump(), depth + 1)
        except Exception:
            return str(value)
    if hasattr(value, "content") and hasattr(value, "type"):
        try:
            return {
                "_object": value.__class__.__name__,
                "type":    to_jsonable(getattr(value, "type",    None), depth + 1),
                "content": to_jsonable(getattr(value, "content", None), depth + 1),
            }
        except Exception:
            return str(value)
    if hasattr(value, "__dict__"):
        try:
            return {"_object": value.__class__.__name__, "data": to_jsonable(vars(value), depth + 1)}
        except Exception:
            return str(value)
    return str(value)


def extract_buffer_snapshot(message_buffer: Optional[Any]) -> Dict[str, Any]:
    if not message_buffer:
        return {}
    return {
        "current_agent":  to_jsonable(message_buffer.current_agent),
        "agent_status":   to_jsonable(message_buffer.agent_status),
        "current_report": to_jsonable(message_buffer.current_report),
        "final_report":   to_jsonable(message_buffer.final_report),
        "messages":       to_jsonable(list(message_buffer.messages)),
        "tool_calls":     to_jsonable(list(message_buffer.tool_calls)),
    }


def build_persistable_session(session_data: Dict[str, Any]) -> Dict[str, Any]:
    persistable = {
        "ticker":               to_jsonable(session_data.get("ticker")),
        "analysis_date":        to_jsonable(session_data.get("analysis_date")),
        "analysts":             to_jsonable(session_data.get("analysts")),
        "alpha_signal":         to_jsonable(session_data.get("alpha_signal")),
        "status":               to_jsonable(session_data.get("status")),
        "current_step":         to_jsonable(session_data.get("current_step")),
        "progress_percent":     to_jsonable(session_data.get("progress_percent")),
        "cancel_requested":     to_jsonable(session_data.get("cancel_requested")),
        "created_at":           to_jsonable(session_data.get("created_at")),
        "decision":             to_jsonable(session_data.get("decision")),
        "final_state":          to_jsonable(session_data.get("final_state")),
        "error":                to_jsonable(session_data.get("error")),
        "error_details":        to_jsonable(session_data.get("error_details")),
    }
    persistable.update(extract_buffer_snapshot(session_data.get("message_buffer")))
    return persistable


# ---------------------------------------------------------------------------
# Các hàm cũ giữ lại để không break import ở nơi khác
# ---------------------------------------------------------------------------

def extract_tool_call_names(message: Any) -> List[str]:
    names: List[str] = []
    tool_calls = (
        message.get("tool_calls") if isinstance(message, dict)
        else getattr(message, "tool_calls", None)
    )
    if not isinstance(tool_calls, list):
        return names
    for call in tool_calls:
        name = (
            call.get("name") or call.get("tool") if isinstance(call, dict)
            else getattr(call, "name", None) or getattr(call, "tool", None)
        )
        if isinstance(name, str) and name:
            names.append(name)
    return names


def to_positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
        return parsed if parsed > 0 else default
    except (TypeError, ValueError):
        return default
import datetime
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


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
                "type": to_jsonable(getattr(value, "type", None), depth + 1),
                "content": to_jsonable(getattr(value, "content", None), depth + 1),
            }
        except Exception:
            return str(value)

    if hasattr(value, "__dict__"):
        try:
            return {
                "_object": value.__class__.__name__,
                "data": to_jsonable(vars(value), depth + 1),
            }
        except Exception:
            return str(value)

    return str(value)


def extract_buffer_snapshot(message_buffer: Optional[Any]) -> Dict[str, Any]:
    if not message_buffer:
        return {}
    return {
        "current_agent": to_jsonable(message_buffer.current_agent),
        "agent_status": to_jsonable(message_buffer.agent_status),
        "current_report": to_jsonable(message_buffer.current_report),
        "final_report": to_jsonable(message_buffer.final_report),
        "messages": to_jsonable(list(message_buffer.messages)),
        "tool_calls": to_jsonable(list(message_buffer.tool_calls)),
    }


def build_persistable_session(session_data: Dict[str, Any]) -> Dict[str, Any]:
    persistable = {
        "ticker": to_jsonable(session_data.get("ticker")),
        "analysis_date": to_jsonable(session_data.get("analysis_date")),
        "analysts": to_jsonable(session_data.get("analysts")),
        "alpha_signal": to_jsonable(session_data.get("alpha_signal")),
        "status": to_jsonable(session_data.get("status")),
        "current_step": to_jsonable(session_data.get("current_step")),
        "progress_percent": to_jsonable(session_data.get("progress_percent")),
        "cancel_requested": to_jsonable(session_data.get("cancel_requested")),
        "created_at": to_jsonable(session_data.get("created_at")),
        "decision": to_jsonable(session_data.get("decision")),
        "decision_raw": to_jsonable(session_data.get("decision_raw")),
        "decision_fused": to_jsonable(session_data.get("decision_fused")),
        "decision_fusion_note": to_jsonable(session_data.get("decision_fusion_note")),
        "final_state": to_jsonable(session_data.get("final_state")),
        "error": to_jsonable(session_data.get("error")),
        "error_details": to_jsonable(session_data.get("error_details")),
    }
    persistable.update(extract_buffer_snapshot(session_data.get("message_buffer")))
    return persistable


def extract_decision_label(raw_decision: Any) -> str:
    text = str(raw_decision or "").upper()
    if "BUY" in text:
        return "BUY"
    if "SELL" in text:
        return "SELL"
    if "HOLD" in text:
        return "HOLD"
    return "HOLD"


def rebuild_reports_from_final_state(final_state: Any) -> Tuple[Optional[str], Optional[str]]:
    if not isinstance(final_state, dict):
        return None, None

    section_titles = {
        "market_report": "Phân tích thị trường",
        "sentiment_report": "Phân tích tâm lý xã hội",
        "news_report": "Phân tích tin tức",
        "fundamentals_report": "Phân tích cơ bản",
        "investment_plan": "Quyết định nhóm nghiên cứu",
        "trader_investment_plan": "Kế hoạch nhóm giao dịch",
        "final_trade_decision": "Quyết định cuối cùng",
    }
    ordered_sections = [
        "market_report",
        "sentiment_report",
        "news_report",
        "fundamentals_report",
        "investment_plan",
        "trader_investment_plan",
        "final_trade_decision",
    ]

    latest_name: Optional[str] = None
    latest_content: Optional[str] = None
    for section_name in ordered_sections:
        section_value = final_state.get(section_name)
        if section_value:
            latest_name = section_name
            latest_content = str(section_value)

    current_report: Optional[str] = None
    if latest_name and latest_content:
        current_title = section_titles.get(latest_name, latest_name)
        current_report = f"### {current_title}\n{latest_content}"

    report_parts: List[str] = []
    if any(final_state.get(name) for name in [
        "market_report", "sentiment_report", "news_report", "fundamentals_report"
    ]):
        report_parts.append("## Báo cáo nhóm phân tích")
        for section_name in ["market_report", "sentiment_report", "news_report", "fundamentals_report"]:
            section_value = final_state.get(section_name)
            if section_value:
                report_parts.append(f"### {section_titles[section_name]}\n{section_value}")

    if final_state.get("investment_plan"):
        report_parts.append("---")
        report_parts.append("## ===== QUYẾT ĐỊNH NHÓM NGHIÊN CỨU =====")
        report_parts.append(str(final_state["investment_plan"]))

    if final_state.get("trader_investment_plan"):
        report_parts.append("---")
        report_parts.append("## ===== KẾ HOẠCH NHÓM GIAO DỊCH =====")
        report_parts.append(str(final_state["trader_investment_plan"]))

    if final_state.get("final_trade_decision"):
        report_parts.append("---")
        report_parts.append("## ===== QUYẾT ĐỊNH CUỐI CÙNG =====")
        report_parts.append(str(final_state["final_trade_decision"]))

    final_report = "\n\n".join(report_parts) if report_parts else None
    return current_report, final_report


def rebuild_agent_status_from_final_state(
    final_state: Any,
    session_status: str,
    default_agent_status: Dict[str, str],
) -> Dict[str, str]:
    if session_status == "completed":
        return {agent: "completed" for agent in default_agent_status}

    if not isinstance(final_state, dict):
        return {}

    status_map = {agent: "pending" for agent in default_agent_status}
    section_to_agent = {
        "market_report": "Market Analyst",
        "sentiment_report": "Social Analyst",
        "news_report": "News Analyst",
        "fundamentals_report": "Fundamentals Analyst",
        "investment_plan": "Research Manager",
        "trader_investment_plan": "Trader",
        "final_trade_decision": "Portfolio Manager",
    }
    for section_name, agent_name in section_to_agent.items():
        if final_state.get(section_name):
            status_map[agent_name] = "completed"

    return status_map


def extract_tool_call_names(message: Any) -> List[str]:
    names: List[str] = []
    tool_calls = None

    if isinstance(message, dict):
        tool_calls = message.get("tool_calls")
    else:
        tool_calls = getattr(message, "tool_calls", None)

    if not isinstance(tool_calls, list):
        return names

    for call in tool_calls:
        if isinstance(call, dict):
            name = call.get("name") or call.get("tool")
        else:
            name = getattr(call, "name", None) or getattr(call, "tool", None)
        if isinstance(name, str) and name:
            names.append(name)

    return names


def to_positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
        return parsed if parsed > 0 else default
    except (TypeError, ValueError):
        return default


def derive_realtime_step(
    chunk: Dict[str, Any],
    max_debate_rounds: int = 1,
    max_risk_rounds: int = 1,
) -> Optional[Tuple[str, int]]:
    max_debate_rounds = to_positive_int(max_debate_rounds, 1)
    max_risk_rounds = to_positive_int(max_risk_rounds, 1)

    if chunk.get("final_trade_decision"):
        return ("✅ Hoàn thành phán quyết cuối cùng", 100)

    risk_state = chunk.get("risk_debate_state")
    if isinstance(risk_state, dict):
        latest_speaker = str(risk_state.get("latest_speaker", ""))
        judge_decision = str(risk_state.get("judge_decision", ""))
        if judge_decision or latest_speaker == "Judge":
            return ("🧑‍⚖️ Risk Judge đang chốt phán quyết cuối", 99)

        max_turns = max(1, 3 * max_risk_rounds)
        turn_count = max(1, to_positive_int(risk_state.get("count"), 1))
        clamped_turn = min(turn_count, max_turns)
        stage_percent = 92 + int((clamped_turn / max_turns) * 6)
        risk_text_map: Dict[str, str] = {
            "Risky": "⚠️ Đang tranh luận: Risky Analyst",
            "Safe": "🛡️ Đang tranh luận: Safe Analyst",
            "Neutral": "⚖️ Đang tranh luận: Neutral Analyst",
        }
        if latest_speaker in risk_text_map:
            return (
                f"{risk_text_map[latest_speaker]} (lượt {clamped_turn}/{max_turns})",
                stage_percent,
            )

    if chunk.get("trader_investment_plan"):
        return ("🧠 Trader đang ra kế hoạch giao dịch", 90)

    if chunk.get("investment_plan"):
        return ("🧑‍⚖️ Research Manager đã chốt phán quyết", 88)

    invest_state = chunk.get("investment_debate_state")
    if isinstance(invest_state, dict):
        current_response = str(invest_state.get("current_response", ""))
        judge_decision = str(invest_state.get("judge_decision", ""))
        if judge_decision:
            return ("🧑‍⚖️ Research Manager đang chốt phán quyết", 87)

        max_turns = max(1, 2 * max_debate_rounds)
        turn_count = max(1, to_positive_int(invest_state.get("count"), 1))
        clamped_turn = min(turn_count, max_turns)
        stage_percent = 80 + int((clamped_turn / max_turns) * 6)
        if current_response.startswith("Bull Analyst"):
            return (f"⚔️ Đang tranh luận: Bull Analyst (lượt {clamped_turn}/{max_turns})", stage_percent)
        if current_response.startswith("Bear Analyst"):
            return (f"⚔️ Đang tranh luận: Bear Analyst (lượt {clamped_turn}/{max_turns})", stage_percent)

    if chunk.get("fundamentals_report"):
        return ("📈 Fundamentals Analyst đã hoàn thành", 76)
    if chunk.get("news_report"):
        return ("📰 News Analyst đã hoàn thành", 70)
    if chunk.get("sentiment_report"):
        return ("💬 Social Analyst đã hoàn thành", 64)
    if chunk.get("market_report"):
        return ("📊 Market Analyst đã hoàn thành", 58)

    messages = chunk.get("messages")
    if isinstance(messages, list) and messages:
        tool_names = extract_tool_call_names(messages[-1])
        if tool_names:
            tool = tool_names[0]
            tool_step_map: Dict[str, Tuple[str, int]] = {
                "get_stock_data": ("⏳ Đang crawl stock data", 15),
                "get_indicators": ("⏳ Đang tính technical indicators", 22),
                "get_news": ("⏳ Đang crawl tin tức và dữ liệu xã hội", 30),
                "get_global_news": ("⏳ Đang tổng hợp tin tức vĩ mô", 38),
                "get_fundamentals": ("⏳ Đang lấy dữ liệu fundamentals", 46),
                "get_balance_sheet": ("⏳ Đang lấy bảng cân đối kế toán", 50),
                "get_cashflow": ("⏳ Đang lấy báo cáo lưu chuyển tiền tệ", 52),
                "get_income_statement": ("⏳ Đang lấy báo cáo kết quả kinh doanh", 54),
                "get_insider_transactions": ("⏳ Đang lấy giao dịch nội bộ", 56),
            }
            if tool in tool_step_map:
                return tool_step_map[tool]

    return None

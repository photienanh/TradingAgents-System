"""FastAPI app entrypoint for TradingAgents."""

import datetime
import asyncio
import os
import time
import logging
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from collections import deque

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG
from dotenv import load_dotenv

from app.routes.market import router as market_router
from app.services.session_serialization import build_persistable_session
from app.services.session_manager import SessionManager
from app.storage.session_store import SQLiteSessionStore

# Alpha integration
from alpha.manager import init_alpha_manager, trigger_if_needed, get_signal_for_ticker, get_status

load_dotenv()

logger = logging.getLogger(__name__)

app = FastAPI(
    title="TradingAgents API",
    description="Multi-Agents LLM Financial Trading Framework",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(market_router)
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# ---------------------------------------------------------------------------
# Session storage — sử dụng SessionManager thay dict + Lock rời
# ---------------------------------------------------------------------------
session_mgr = SessionManager()  # Thread-safe

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SESSION_DB_PATH = PROJECT_ROOT / "app" / "data" / "sessions.db"
LEGACY_SESSION_JSON_PATH = PROJECT_ROOT / "app" / "data" / "sessions.json"
session_store = SQLiteSessionStore(SESSION_DB_PATH)


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def _save_sessions_to_disk() -> None:
    persistable = {
        sid: build_persistable_session(data)
        for sid, data in session_mgr.all_items()
    }
    session_store.save_all(persistable)


def _load_sessions_from_disk() -> None:
    session_store.migrate_from_json_file(LEGACY_SESSION_JSON_PATH)
    try:
        sessions = session_store.load_all()
        for sid, data in sessions.items():
            if data.get("status") in {"initializing", "running"}:
                data["status"] = "error"
                data["error"] = "Session interrupted because server restarted"
            session_mgr.set(sid, data)
    except Exception as exc:
        logger.warning("Không load được sessions từ sqlite: %s", exc)


# ---------------------------------------------------------------------------
# Alpha signal loader — dùng SignalStore thay CSV tĩnh
# ---------------------------------------------------------------------------

def _load_alphagpt_signal(ticker: str) -> Dict[str, Any]:
    """Đọc signal từ SignalStore (được cập nhật bởi daily pipeline)."""
    return get_signal_for_ticker(ticker)


# ---------------------------------------------------------------------------
# Decision helpers
# ---------------------------------------------------------------------------

def _extract_decision_label(raw_decision: Any) -> str:
    text = str(raw_decision or "").upper()
    if "BUY" in text:
        return "BUY"
    if "SELL" in text:
        return "SELL"
    if "HOLD" in text:
        return "HOLD"
    return "HOLD"


def _fuse_decision_with_alphagpt(ta_decision: str, alpha_side: Optional[str]) -> Tuple[str, str]:
    decision = ta_decision if ta_decision in {"BUY", "SELL", "HOLD"} else "HOLD"
    side = (alpha_side or "neutral").strip().lower()
    if side == "long":
        if decision == "HOLD":
            return "BUY", "AlphaGPT long bias upgraded HOLD to BUY"
        if decision == "SELL":
            return "HOLD", "Conflict between TA SELL and AlphaGPT long; downgraded to HOLD"
        return "BUY", "TA BUY aligned with AlphaGPT long"
    if side == "short":
        if decision == "HOLD":
            return "SELL", "AlphaGPT short bias downgraded HOLD to SELL"
        if decision == "BUY":
            return "HOLD", "Conflict between TA BUY and AlphaGPT short; downgraded to HOLD"
        return "SELL", "TA SELL aligned with AlphaGPT short"
    return decision, "No AlphaGPT directional bias (neutral or unavailable)"


def _rebuild_reports_from_final_state(final_state: Any) -> Tuple[Optional[str], Optional[str]]:
    if not isinstance(final_state, dict):
        return None, None

    section_titles = {
        "market_report": "Phân tích thị trường",
        "sentiment_report": "Phân tích tâm lý xã hội",
        "news_report": "Phân tích tin tức",
        "fundamentals_report": "Phân tích cơ bản",
        "quant_report": "Phân tích định lượng (AlphaGPT)",
        "investment_plan": "Quyết định nhóm nghiên cứu",
        "trader_investment_plan": "Kế hoạch nhóm giao dịch",
        "final_trade_decision": "Quyết định cuối cùng",
    }
    ordered_sections = [
        "market_report", "sentiment_report", "news_report", "fundamentals_report", "quant_report",
        "investment_plan", "trader_investment_plan", "final_trade_decision",
    ]

    latest_name = latest_content = None
    for sec in ordered_sections:
        val = final_state.get(sec)
        if val:
            latest_name, latest_content = sec, str(val)

    current_report = None
    if latest_name and latest_content:
        current_report = f"### {section_titles.get(latest_name, latest_name)}\n{latest_content}"

    report_parts: List[str] = []
    if any(final_state.get(n) for n in ["market_report", "sentiment_report", "news_report", "fundamentals_report", "quant_report"]):
        report_parts.append("## Báo cáo nhóm phân tích")
        for sec in ["market_report", "sentiment_report", "news_report", "fundamentals_report", "quant_report"]:
            if final_state.get(sec):
                report_parts.append(f"### {section_titles[sec]}\n{final_state[sec]}")
    if final_state.get("investment_plan"):
        report_parts += ["---", "## ===== QUYẾT ĐỊNH NHÓM NGHIÊN CỨU =====", str(final_state["investment_plan"])]
    if final_state.get("trader_investment_plan"):
        report_parts += ["---", "## ===== KẾ HOẠCH NHÓM GIAO DỊCH =====", str(final_state["trader_investment_plan"])]
    if final_state.get("final_trade_decision"):
        report_parts += ["---", "## ===== QUYẾT ĐỊNH CUỐI CÙNG =====", str(final_state["final_trade_decision"])]

    return current_report, "\n\n".join(report_parts) if report_parts else None


def _rebuild_agent_status_from_final_state(final_state: Any, session_status: str) -> Dict[str, str]:
    default_agents = {a: "pending" for a in MessageBuffer().agent_status}
    if session_status == "completed":
        return {a: "completed" for a in default_agents}
    if not isinstance(final_state, dict):
        return {}

    status_map = dict(default_agents)
    section_to_agent = {
        "market_report": "Market Analyst",
        "sentiment_report": "Social Analyst",
        "news_report": "News Analyst",
        "fundamentals_report": "Fundamentals Analyst",
        "quant_report": "AlphaGPT Analyst",
        "investment_plan": "Research Manager",
        "trader_investment_plan": "Trader",
        "final_trade_decision": "Portfolio Manager",
    }
    for sec, agent in section_to_agent.items():
        if final_state.get(sec):
            status_map[agent] = "completed"
    return status_map


def _extract_tool_call_names(message: Any) -> List[str]:
    names: List[str] = []
    tool_calls = message.get("tool_calls") if isinstance(message, dict) else getattr(message, "tool_calls", None)
    if not isinstance(tool_calls, list):
        return names
    for call in tool_calls:
        name = call.get("name") or call.get("tool") if isinstance(call, dict) else getattr(call, "name", None)
        if isinstance(name, str) and name:
            names.append(name)
    return names


def _to_positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
        return parsed if parsed > 0 else default
    except (TypeError, ValueError):
        return default


def _derive_realtime_step(chunk, max_debate_rounds=1, max_risk_rounds=1):
    max_debate_rounds = _to_positive_int(max_debate_rounds, 1)
    max_risk_rounds = _to_positive_int(max_risk_rounds, 1)

    if chunk.get("final_trade_decision"):
        return ("✅ Hoàn thành phán quyết cuối cùng", 100)

    risk_state = chunk.get("risk_debate_state")
    if isinstance(risk_state, dict):
        latest_speaker = str(risk_state.get("latest_speaker", ""))
        judge_decision = str(risk_state.get("judge_decision", ""))
        if judge_decision or latest_speaker == "Judge":
            return ("🧑‍⚖️ Risk Judge đang chốt phán quyết cuối", 99)
        max_turns = max(1, 3 * max_risk_rounds)
        turn_count = max(1, _to_positive_int(risk_state.get("count"), 1))
        clamped_turn = min(turn_count, max_turns)
        stage_percent = 92 + int((clamped_turn / max_turns) * 6)
        risk_text_map = {
            "Risky": "⚠️ Đang tranh luận: Risky Analyst",
            "Safe": "🛡️ Đang tranh luận: Safe Analyst",
            "Neutral": "⚖️ Đang tranh luận: Neutral Analyst",
        }
        if latest_speaker in risk_text_map:
            return (f"{risk_text_map[latest_speaker]} (lượt {clamped_turn}/{max_turns})", stage_percent)

    if chunk.get("trader_investment_plan"):
        return ("🧠 Trader đang ra kế hoạch giao dịch", 90)
    if chunk.get("investment_plan"):
        return ("🧑‍⚖️ Research Manager đã chốt phán quyết", 88)

    invest_state = chunk.get("investment_debate_state")
    if isinstance(invest_state, dict):
        current_response = str(invest_state.get("current_response", ""))
        if str(invest_state.get("judge_decision", "")):
            return ("🧑‍⚖️ Research Manager đang chốt phán quyết", 87)
        max_turns = max(1, 2 * max_debate_rounds)
        turn_count = max(1, _to_positive_int(invest_state.get("count"), 1))
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
        tool_names = _extract_tool_call_names(messages[-1])
        if tool_names:
            tool_step_map = {
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
            if tool_names[0] in tool_step_map:
                return tool_step_map[tool_names[0]]

    return None


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class AnalysisRequest(BaseModel):
    ticker: str
    analysis_date: Optional[str] = None
    analysts: Optional[List[str]] = ["market", "social", "news", "fundamentals"]
    research_depth: Optional[int] = 1
    deep_think_llm: Optional[str] = "gpt-4o-mini"
    quick_think_llm: Optional[str] = "gpt-4o-mini"
    max_debate_rounds: Optional[int] = 1
    data_vendors: Optional[Dict[str, str]] = Field(
        default_factory=lambda: DEFAULT_CONFIG["data_vendors"].copy()
    )


class AnalysisResponse(BaseModel):
    session_id: str
    status: str
    message: str


class AnalysisStatus(BaseModel):
    session_id: str
    status: str
    current_step: Optional[str] = None
    progress_percent: Optional[int] = None
    current_agent: Optional[str] = None
    agent_status: Dict[str, str]
    current_report: Optional[str] = None
    final_report: Optional[str] = None
    decision: Optional[Any] = None
    decision_raw: Optional[Any] = None
    decision_fused: Optional[str] = None
    decision_fusion_note: Optional[str] = None
    alpha_signal: Optional[Dict[str, Any]] = None
    messages: List[Dict[str, Any]] = []
    tool_calls: List[Dict[str, Any]] = []
    error: Optional[str] = None
    error_details: Optional[str] = None


# ---------------------------------------------------------------------------
# MessageBuffer
# ---------------------------------------------------------------------------

class MessageBuffer:
    def __init__(self, max_length=100):
        self.messages = deque(maxlen=max_length)
        self.tool_calls = deque(maxlen=max_length)
        self.current_report = None
        self.final_report = None
        self.agent_status = {
            "Market Analyst": "pending",
            "Social Analyst": "pending",
            "News Analyst": "pending",
            "Fundamentals Analyst": "pending",
            "Bull Researcher": "pending",
            "Bear Researcher": "pending",
            "Research Manager": "pending",
            "Trader": "pending",
            "AlphaGPT Analyst": "pending",
            "Risky Analyst": "pending",
            "Safe Analyst": "pending",
            "Neutral Analyst": "pending",
            "Portfolio Manager": "pending",
        }
        self.current_agent = None
        self.report_sections = {
            "market_report": None,
            "sentiment_report": None,
            "news_report": None,
            "fundamentals_report": None,
            "quant_report": None,
            "investment_plan": None,
            "trader_investment_plan": None,
            "final_trade_decision": None,
        }

    def add_message(self, message_type, content):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.messages.append({"timestamp": timestamp, "type": message_type, "content": content})

    def add_tool_call(self, tool_name, args):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.tool_calls.append({"timestamp": timestamp, "tool": tool_name, "args": args})

    def update_agent_status(self, agent, status):
        if agent in self.agent_status:
            self.agent_status[agent] = status
            self.current_agent = agent

    def update_report_section(self, section_name, content):
        if section_name in self.report_sections:
            self.report_sections[section_name] = content
            self._update_current_report()

    def _update_current_report(self):
        section_titles = {
            "market_report": "Phân tích thị trường",
            "sentiment_report": "Phân tích tâm lý xã hội",
            "news_report": "Phân tích tin tức",
            "fundamentals_report": "Phân tích cơ bản",
            "quant_report": "Phân tích định lượng (AlphaGPT)",
            "investment_plan": "Quyết định nhóm nghiên cứu",
            "trader_investment_plan": "Kế hoạch nhóm giao dịch",
            "final_trade_decision": "Quyết định cuối cùng",
        }
        for sec, content in self.report_sections.items():
            if content is not None:
                self.current_report = f"### {section_titles[sec]}\n{content}"
        self._update_final_report()

    def _update_final_report(self):
        parts = []
        section_titles = {
            "market_report": "Phân tích thị trường",
            "sentiment_report": "Phân tích tâm lý xã hội",
            "news_report": "Phân tích tin tức",
            "fundamentals_report": "Phân tích cơ bản",
            "quant_report": "Phân tích định lượng (AlphaGPT)",
        }
        if any(self.report_sections[s] for s in section_titles):
            parts.append("## Báo cáo nhóm phân tích")
            for sec, title in section_titles.items():
                if self.report_sections[sec]:
                    parts.append(f"### {title}\n{self.report_sections[sec]}")
        for key, label in [
            ("investment_plan", "## ===== QUYẾT ĐỊNH NHÓM NGHIÊN CỨU ====="),
            ("trader_investment_plan", "## ===== KẾ HOẠCH NHÓM GIAO DỊCH ====="),
            ("final_trade_decision", "## ===== QUYẾT ĐỊNH CUỐI CÙNG ====="),
        ]:
            if self.report_sections[key]:
                parts += ["---", label, str(self.report_sections[key])]
        self.final_report = "\n\n".join(parts) if parts else None


# ---------------------------------------------------------------------------
# Background analysis task
# ---------------------------------------------------------------------------

async def run_trading_analysis(
    session_id: str,
    ticker: str,
    analysis_date: str,
    config: dict,
    analysts: List[str],
    alpha_signal: Optional[Dict[str, Any]] = None,
):
    message_buffer = MessageBuffer()
    session_mgr.update(session_id, {
        "message_buffer": message_buffer,
        "status": "running",
        "current_step": "Đang khởi tạo đồ thị phân tích",
        "progress_percent": 10,
        "cancel_requested": False,
    })
    _save_sessions_to_disk()

    try:
        if session_mgr.get_field(session_id, "cancel_requested"):
            raise asyncio.CancelledError("Analysis cancelled before start")

        graph = TradingAgentsGraph(debug=True, config=config, selected_analysts=list(analysts))

        analyst_map = {
            "market": "Market Analyst",
            "social": "Social Analyst",
            "news": "News Analyst",
            "fundamentals": "Fundamentals Analyst",
        }
        for key in analysts:
            if key in analyst_map:
                message_buffer.update_agent_status(analyst_map[key], "in_progress")
        for key, name in analyst_map.items():
            if key not in analysts:
                message_buffer.agent_status[name] = "not_selected"

        max_debate_rounds = _to_positive_int(config.get("max_debate_rounds"), 1)
        max_risk_rounds = _to_positive_int(config.get("max_risk_discuss_rounds"), 1)
        last_progress_persist = time.monotonic()

        def _on_graph_progress(chunk: Dict[str, Any]) -> None:
            nonlocal last_progress_persist
            if session_mgr.get_field(session_id, "cancel_requested"):
                raise asyncio.CancelledError("Analysis cancelled by user")

            progress_info = _derive_realtime_step(chunk, max_debate_rounds, max_risk_rounds)
            if progress_info is None:
                return

            step_text, step_percent = progress_info
            prev_pct = int(session_mgr.get_field(session_id, "progress_percent") or 0)
            session_mgr.update(session_id, {
                "current_step": step_text,
                "progress_percent": max(prev_pct, step_percent),
            })

            report_by_analyst = {
                "market": "market_report",
                "social": "sentiment_report",
                "news": "news_report",
                "fundamentals": "fundamentals_report",
            }
            first_incomplete = False
            for key in analysts:
                name = analyst_map.get(key)
                rkey = report_by_analyst.get(key)
                if not name or not rkey:
                    continue
                if chunk.get(rkey):
                    message_buffer.update_agent_status(name, "completed")
                    continue
                if not first_incomplete:
                    message_buffer.update_agent_status(name, "in_progress")
                    first_incomplete = True
                else:
                    message_buffer.update_agent_status(name, "pending")

            invest_state = chunk.get("investment_debate_state")
            if isinstance(invest_state, dict):
                cr = str(invest_state.get("current_response", ""))
                jd = str(invest_state.get("judge_decision", ""))
                if cr.startswith("Bull Analyst"):
                    message_buffer.update_agent_status("Bull Researcher", "in_progress")
                elif cr.startswith("Bear Analyst"):
                    message_buffer.update_agent_status("Bull Researcher", "completed")
                    message_buffer.update_agent_status("Bear Researcher", "in_progress")
                if jd:
                    for a in ["Bull Researcher", "Bear Researcher", "Research Manager"]:
                        message_buffer.update_agent_status(a, "completed")

            if chunk.get("investment_plan") and message_buffer.agent_status.get("Research Manager") != "completed":
                message_buffer.update_agent_status("Research Manager", "in_progress")
            if chunk.get("trader_investment_plan"):
                message_buffer.update_agent_status("Trader", "completed")
            elif chunk.get("investment_plan"):
                message_buffer.update_agent_status("Trader", "in_progress")

            risk_state = chunk.get("risk_debate_state")
            if isinstance(risk_state, dict):
                ls = str(risk_state.get("latest_speaker", ""))
                if ls == "AlphaGPT":
                    message_buffer.update_agent_status("AlphaGPT Analyst", "in_progress")
                elif ls == "Risky":
                    message_buffer.update_agent_status("AlphaGPT Analyst", "completed")
                    message_buffer.update_agent_status("Risky Analyst", "in_progress")
                elif ls == "Safe":
                    message_buffer.update_agent_status("Risky Analyst", "completed")
                    message_buffer.update_agent_status("Safe Analyst", "in_progress")
                elif ls == "Neutral":
                    message_buffer.update_agent_status("Safe Analyst", "completed")
                    message_buffer.update_agent_status("Neutral Analyst", "in_progress")
                elif ls == "Judge":
                    for a in ["Risky Analyst", "Safe Analyst", "Neutral Analyst"]:
                        message_buffer.update_agent_status(a, "completed")
                    message_buffer.update_agent_status("Portfolio Manager", "in_progress")

            if chunk.get("final_trade_decision"):
                message_buffer.update_agent_status("Portfolio Manager", "completed")

            now = time.monotonic()
            if now - last_progress_persist >= 1.0:
                _save_sessions_to_disk()
                last_progress_persist = now

        def _run_graph():
            return graph.propagate(
                ticker,
                analysis_date,
                alphagpt_signal=alpha_signal,
                progress_callback=_on_graph_progress,
            )

        final_state, decision = await asyncio.to_thread(_run_graph)

        alpha_note = "AlphaGPT signal routed through AlphaGPT Analyst node in decision layer"
        if isinstance(alpha_signal, dict) and not alpha_signal.get("enabled", False):
            alpha_note = "AlphaGPT Analyst active but no directional signal available"

        section_map = {
            "market_report": "Market Analyst",
            "sentiment_report": "Social Analyst",
            "news_report": "News Analyst",
            "fundamentals_report": "Fundamentals Analyst",
            "quant_report": "AlphaGPT Analyst",
        }
        for sec_key, analyst_name in section_map.items():
            if final_state.get(sec_key):
                message_buffer.update_report_section(sec_key, final_state[sec_key])
                message_buffer.update_agent_status(analyst_name, "completed")

        if final_state.get("investment_plan"):
            for a in ["Bull Researcher", "Bear Researcher"]:
                message_buffer.update_agent_status(a, "completed")
            message_buffer.update_report_section("investment_plan", final_state["investment_plan"])
            message_buffer.update_agent_status("Research Manager", "completed")

        if final_state.get("trader_investment_plan"):
            message_buffer.update_report_section("trader_investment_plan", final_state["trader_investment_plan"])
            message_buffer.update_agent_status("Trader", "completed")

        message_buffer.update_agent_status("AlphaGPT Analyst", "completed")

        if final_state.get("final_trade_decision"):
            for a in ["Risky Analyst", "Safe Analyst", "Neutral Analyst"]:
                message_buffer.update_agent_status(a, "completed")
            message_buffer.update_report_section("final_trade_decision", str(final_state["final_trade_decision"]))
            message_buffer.update_agent_status("Portfolio Manager", "completed")

        for agent, status in list(message_buffer.agent_status.items()):
            if status not in {"not_selected", "completed"}:
                message_buffer.update_agent_status(agent, "completed")

        ta_decision_label = _extract_decision_label(decision)
        alpha_side = alpha_signal.get("side") if isinstance(alpha_signal, dict) else None
        fused_decision, fusion_note = _fuse_decision_with_alphagpt(ta_decision_label, alpha_side)

        session_mgr.update(session_id, {
            "status": "completed",
            "current_step": "✅ Hoàn thành phân tích",
            "progress_percent": 100,
            "decision": decision,
            "decision_raw": decision,
            "decision_fused": fused_decision,
            "decision_fusion_note": fusion_note,
            "alpha_signal": alpha_signal,
            "final_state": final_state,
        })
        _save_sessions_to_disk()

    except asyncio.CancelledError as e:
        session_mgr.update(session_id, {
            "status": "cancelled",
            "current_step": "🛑 Đã hủy phân tích theo yêu cầu",
            "error": str(e),
            "error_details": None,
        })
        if not session_mgr.get_field(session_id, "progress_percent"):
            session_mgr.set_field(session_id, "progress_percent", 0)
        _save_sessions_to_disk()

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logger.error("Lỗi phân tích %s: %s", session_id, error_details)
        session_mgr.update(session_id, {
            "status": "error",
            "current_step": "❌ Phân tích thất bại",
            "error": str(e),
            "error_details": error_details,
        })
        if not session_mgr.get_field(session_id, "progress_percent"):
            session_mgr.set_field(session_id, "progress_percent", 0)
        mb = session_mgr.get_field(session_id, "message_buffer")
        if mb:
            mb.add_message("Error", f"Analysis failed: {str(e)}")
        _save_sessions_to_disk()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/market", response_class=HTMLResponse)
async def market_page(request: Request):
    return templates.TemplateResponse("market_new.html", {"request": request})


@app.post("/api/analyze", response_model=AnalysisResponse)
async def start_analysis(request: AnalysisRequest, background_tasks: BackgroundTasks):
    try:
        selected_analysts = request.analysts or ["market", "social", "news", "fundamentals"]
        session_id = f"{request.ticker}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Trigger alpha pipeline nếu cần (non-blocking)
        trigger_if_needed()

        alpha_signal = _load_alphagpt_signal(request.ticker)
        analysis_date = request.analysis_date or datetime.datetime.now().strftime("%Y-%m-%d")

        config = DEFAULT_CONFIG.copy()
        config["deep_think_llm"] = request.deep_think_llm
        config["quick_think_llm"] = request.quick_think_llm
        config["max_debate_rounds"] = request.research_depth or request.max_debate_rounds
        config["data_vendors"] = request.data_vendors

        session_mgr.set(session_id, {
            "ticker": request.ticker,
            "analysis_date": analysis_date,
            "analysts": selected_analysts,
            "alpha_signal": alpha_signal,
            "status": "initializing",
            "current_step": "Đang tạo phiên phân tích",
            "progress_percent": 5,
            "cancel_requested": False,
            "created_at": datetime.datetime.now().isoformat(),
        })
        _save_sessions_to_disk()

        background_tasks.add_task(
            run_trading_analysis,
            session_id, request.ticker, analysis_date, config, selected_analysts, alpha_signal,
        )
        return AnalysisResponse(session_id=session_id, status="started", message=f"Analysis started for {request.ticker}")

    except Exception as e:
        import traceback
        logger.error("Lỗi start_analysis: %s", traceback.format_exc())
        return JSONResponse(status_code=500, content={"error": str(e), "details": traceback.format_exc()})


@app.get("/api/status/{session_id}", response_model=AnalysisStatus)
async def get_analysis_status(session_id: str):
    session = session_mgr.get(session_id)
    if session is None:
        return JSONResponse(status_code=404, content={"error": "Session not found"})

    mb = session.get("message_buffer")
    if mb:
        return AnalysisStatus(
            session_id=session_id,
            status=session["status"],
            current_step=session.get("current_step"),
            progress_percent=session.get("progress_percent"),
            current_agent=mb.current_agent,
            agent_status=mb.agent_status,
            current_report=mb.current_report,
            final_report=mb.final_report,
            decision=session.get("decision"),
            decision_raw=session.get("decision_raw"),
            decision_fused=session.get("decision_fused"),
            decision_fusion_note=session.get("decision_fusion_note"),
            alpha_signal=session.get("alpha_signal"),
            messages=list(mb.messages),
            tool_calls=list(mb.tool_calls),
            error=session.get("error"),
            error_details=session.get("error_details"),
        )

    # Session từ disk
    final_state = session.get("final_state")
    rebuilt_current, rebuilt_final = _rebuild_reports_from_final_state(final_state)
    loaded_agent_status = session.get("agent_status") or _rebuild_agent_status_from_final_state(
        final_state, str(session.get("status", ""))
    )

    return AnalysisStatus(
        session_id=session_id,
        status=session["status"],
        current_step=session.get("current_step"),
        progress_percent=session.get("progress_percent"),
        current_agent=session.get("current_agent"),
        agent_status=loaded_agent_status or {},
        current_report=session.get("current_report") or rebuilt_current,
        final_report=session.get("final_report") or rebuilt_final,
        decision=session.get("decision"),
        decision_raw=session.get("decision_raw"),
        decision_fused=session.get("decision_fused"),
        decision_fusion_note=session.get("decision_fusion_note"),
        alpha_signal=session.get("alpha_signal"),
        messages=session.get("messages") or [],
        tool_calls=session.get("tool_calls") or [],
        error=session.get("error"),
        error_details=session.get("error_details"),
    )


@app.get("/api/sessions")
async def list_sessions():
    sessions = [
        {
            "session_id": sid,
            "ticker": data.get("ticker"),
            "analysis_date": data.get("analysis_date"),
            "status": data.get("status"),
            "created_at": data.get("created_at"),
        }
        for sid, data in session_mgr.all_items()
    ]
    return {"sessions": sessions}


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    if session_mgr.delete(session_id):
        _save_sessions_to_disk()
        return {"message": "Session deleted successfully"}
    return JSONResponse(status_code=404, content={"error": "Session not found"})


@app.post("/api/sessions/{session_id}/cancel")
async def cancel_session(session_id: str):
    session = session_mgr.get(session_id)
    if session is None:
        return JSONResponse(status_code=404, content={"error": "Session not found"})
    status = str(session.get("status", ""))
    if status in {"completed", "error", "cancelled"}:
        return {"message": f"Session is already {status}", "status": status}
    session_mgr.update(session_id, {
        "cancel_requested": True,
        "current_step": "🛑 Đang hủy phân tích...",
    })
    _save_sessions_to_disk()
    return {"message": "Cancellation requested", "status": session.get("status")}
"""
app/main.py

Changelog so với phiên bản trước:
- Xóa _rebuild_reports_from_final_state() và _rebuild_agent_status_from_final_state()
  → import từ app.services.session_serialization (single source of truth)
- Fix bug agent status: analyst đã completed không bị reset về in_progress
- Import SECTION_TITLES, DEFAULT_AGENT_NAMES từ session_serialization
"""

import datetime
import asyncio
import os
import time
import logging
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from collections import deque

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG
from dotenv import load_dotenv

from app.routes.market import router as market_router
from app.services.session_serialization import (
    build_persistable_session,
    rebuild_reports_from_final_state,
    rebuild_agent_status_from_final_state,
    SECTION_TITLES,
    DEFAULT_AGENT_NAMES,
)
from app.services.session_manager import SessionManager
from app.storage.session_store import SQLiteSessionStore

# Alpha integration
from alpha.manager import init_alpha_manager, trigger_if_needed, get_signal_for_ticker, get_status

load_dotenv()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="TradingAgents API",
    description="Multi-Agents LLM Financial Trading Framework",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(market_router)
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# ---------------------------------------------------------------------------
# Session storage
# ---------------------------------------------------------------------------
session_mgr = SessionManager()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SESSION_DB_PATH = PROJECT_ROOT / "app" / "data" / "sessions.db"
LEGACY_SESSION_JSON_PATH = PROJECT_ROOT / "app" / "data" / "sessions.json"
session_store = SQLiteSessionStore(SESSION_DB_PATH)


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def _save_sessions_to_disk() -> None:
    persistable = {
        sid: build_persistable_session(data)
        for sid, data in session_mgr.all_items()
    }
    session_store.save_all(persistable)


def _load_sessions_from_disk() -> None:
    session_store.migrate_from_json_file(LEGACY_SESSION_JSON_PATH)
    try:
        sessions = session_store.load_all()
        for sid, data in sessions.items():
            if data.get("status") in {"initializing", "running"}:
                data["status"] = "error"
                data["error"] = "Session interrupted because server restarted"
            session_mgr.set(sid, data)
    except Exception as exc:
        logger.warning("Không load được sessions từ sqlite: %s", exc)


# ---------------------------------------------------------------------------
# Alpha signal loader
# ---------------------------------------------------------------------------

def _load_alphagpt_signal(ticker: str) -> Dict[str, Any]:
    return get_signal_for_ticker(ticker)


# ---------------------------------------------------------------------------
# Decision helpers
# ---------------------------------------------------------------------------

def _extract_decision_label(raw_decision: Any) -> str:
    text = str(raw_decision or "").upper()
    if "BUY"  in text: return "BUY"
    if "SELL" in text: return "SELL"
    if "HOLD" in text: return "HOLD"
    return "HOLD"


def _fuse_decision_with_alphagpt(
    ta_decision: str, alpha_side: Optional[str]
) -> Tuple[str, str]:
    decision = ta_decision if ta_decision in {"BUY", "SELL", "HOLD"} else "HOLD"
    side = (alpha_side or "neutral").strip().lower()
    if side == "long":
        if decision == "HOLD":  return "BUY",  "AlphaGPT long bias upgraded HOLD to BUY"
        if decision == "SELL":  return "HOLD", "Conflict between TA SELL and AlphaGPT long; downgraded to HOLD"
        return "BUY",  "TA BUY aligned with AlphaGPT long"
    if side == "short":
        if decision == "HOLD":  return "SELL", "AlphaGPT short bias downgraded HOLD to SELL"
        if decision == "BUY":   return "HOLD", "Conflict between TA BUY and AlphaGPT short; downgraded to HOLD"
        return "SELL", "TA SELL aligned with AlphaGPT short"
    return decision, "No AlphaGPT directional bias (neutral or unavailable)"


def _extract_tool_call_names(message: Any) -> List[str]:
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
            else getattr(call, "name", None)
        )
        if isinstance(name, str) and name:
            names.append(name)
    return names


def _to_positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
        return parsed if parsed > 0 else default
    except (TypeError, ValueError):
        return default


def _derive_realtime_step(chunk, max_debate_rounds=1, max_risk_rounds=1):
    max_debate_rounds = _to_positive_int(max_debate_rounds, 1)
    max_risk_rounds   = _to_positive_int(max_risk_rounds, 1)

    if chunk.get("final_trade_decision"):
        return ("✅ Hoàn thành phán quyết cuối cùng", 100)

    risk_state = chunk.get("risk_debate_state")
    if isinstance(risk_state, dict):
        latest_speaker = str(risk_state.get("latest_speaker", ""))
        judge_decision = str(risk_state.get("judge_decision", ""))
        if judge_decision or latest_speaker == "Judge":
            return ("🧑‍⚖️ Risk Judge đang chốt phán quyết cuối", 99)
        max_turns    = max(1, 3 * max_risk_rounds)
        turn_count   = max(1, _to_positive_int(risk_state.get("count"), 1))
        clamped_turn = min(turn_count, max_turns)
        stage_pct    = 92 + int((clamped_turn / max_turns) * 6)
        risk_text_map = {
            "Risky":   "⚠️ Đang tranh luận: Risky Analyst",
            "Safe":    "🛡️ Đang tranh luận: Safe Analyst",
            "Neutral": "⚖️ Đang tranh luận: Neutral Analyst",
        }
        if latest_speaker in risk_text_map:
            return (f"{risk_text_map[latest_speaker]} (lượt {clamped_turn}/{max_turns})", stage_pct)

    if chunk.get("trader_investment_plan"):
        return ("🧠 Trader đang ra kế hoạch giao dịch", 90)
    if chunk.get("investment_plan"):
        return ("🧑‍⚖️ Research Manager đã chốt phán quyết", 88)

    invest_state = chunk.get("investment_debate_state")
    if isinstance(invest_state, dict):
        current_response = str(invest_state.get("current_response", ""))
        if str(invest_state.get("judge_decision", "")):
            return ("🧑‍⚖️ Research Manager đang chốt phán quyết", 87)
        max_turns    = max(1, 2 * max_debate_rounds)
        turn_count   = max(1, _to_positive_int(invest_state.get("count"), 1))
        clamped_turn = min(turn_count, max_turns)
        stage_pct    = 80 + int((clamped_turn / max_turns) * 6)
        if current_response.startswith("Bull Analyst"):
            return (f"⚔️ Đang tranh luận: Bull Analyst (lượt {clamped_turn}/{max_turns})", stage_pct)
        if current_response.startswith("Bear Analyst"):
            return (f"⚔️ Đang tranh luận: Bear Analyst (lượt {clamped_turn}/{max_turns})", stage_pct)

    if chunk.get("fundamentals_report"): return ("📈 Fundamentals Analyst đã hoàn thành", 76)
    if chunk.get("news_report"):         return ("📰 News Analyst đã hoàn thành", 70)
    if chunk.get("sentiment_report"):    return ("💬 Social Analyst đã hoàn thành", 64)
    if chunk.get("market_report"):       return ("📊 Market Analyst đã hoàn thành", 58)

    messages = chunk.get("messages")
    if isinstance(messages, list) and messages:
        tool_names = _extract_tool_call_names(messages[-1])
        if tool_names:
            tool_step_map = {
                "get_stock_data":           ("⏳ Đang crawl stock data", 15),
                "get_indicators":           ("⏳ Đang tính technical indicators", 22),
                "get_news":                 ("⏳ Đang crawl tin tức và dữ liệu xã hội", 30),
                "get_global_news":          ("⏳ Đang tổng hợp tin tức vĩ mô", 38),
                "get_fundamentals":         ("⏳ Đang lấy dữ liệu fundamentals", 46),
                "get_balance_sheet":        ("⏳ Đang lấy bảng cân đối kế toán", 50),
                "get_cashflow":             ("⏳ Đang lấy báo cáo lưu chuyển tiền tệ", 52),
                "get_income_statement":     ("⏳ Đang lấy báo cáo kết quả kinh doanh", 54),
                "get_insider_transactions": ("⏳ Đang lấy giao dịch nội bộ", 56),
            }
            if tool_names[0] in tool_step_map:
                return tool_step_map[tool_names[0]]
    return None


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class AnalysisRequest(BaseModel):
    ticker: str
    analysis_date: Optional[str] = None
    analysts: Optional[List[str]] = ["market", "social", "news", "fundamentals"]
    research_depth: Optional[int] = 1
    deep_think_llm: Optional[str] = "gpt-4o-mini"
    quick_think_llm: Optional[str] = "gpt-4o-mini"
    max_debate_rounds: Optional[int] = 1
    data_vendors: Optional[Dict[str, str]] = Field(
        default_factory=lambda: DEFAULT_CONFIG["data_vendors"].copy()
    )


class AnalysisResponse(BaseModel):
    session_id: str
    status: str
    message: str


class AnalysisStatus(BaseModel):
    session_id: str
    status: str
    current_step: Optional[str] = None
    progress_percent: Optional[int] = None
    current_agent: Optional[str] = None
    agent_status: Dict[str, str]
    current_report: Optional[str] = None
    final_report: Optional[str] = None
    decision: Optional[Any] = None
    decision_raw: Optional[Any] = None
    decision_fused: Optional[str] = None
    decision_fusion_note: Optional[str] = None
    alpha_signal: Optional[Dict[str, Any]] = None
    messages: List[Dict[str, Any]] = []
    tool_calls: List[Dict[str, Any]] = []
    error: Optional[str] = None
    error_details: Optional[str] = None


# ---------------------------------------------------------------------------
# MessageBuffer
# ---------------------------------------------------------------------------

class MessageBuffer:
    def __init__(self, max_length=100):
        self.messages     = deque(maxlen=max_length)
        self.tool_calls   = deque(maxlen=max_length)
        self.current_report = None
        self.final_report   = None
        # Dùng DEFAULT_AGENT_NAMES từ session_serialization — single source of truth
        self.agent_status = {a: "pending" for a in DEFAULT_AGENT_NAMES}
        self.current_agent = None
        self.report_sections = {k: None for k in SECTION_TITLES}

    def add_message(self, message_type, content):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self.messages.append({"timestamp": ts, "type": message_type, "content": content})

    def add_tool_call(self, tool_name, args):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self.tool_calls.append({"timestamp": ts, "tool": tool_name, "args": args})

    def update_agent_status(self, agent: str, status: str) -> None:
        if agent in self.agent_status:
            # Không bao giờ downgrade từ completed → in_progress/pending
            if self.agent_status[agent] == "completed" and status != "completed":
                return
            self.agent_status[agent] = status
            if status == "in_progress":
                self.current_agent = agent

    def update_report_section(self, section_name: str, content: str) -> None:
        if section_name in self.report_sections:
            self.report_sections[section_name] = content
            self._update_current_report()

    def _update_current_report(self):
        for sec, content in self.report_sections.items():
            if content is not None:
                title = SECTION_TITLES.get(sec, sec)
                self.current_report = f"### {title}\n{content}"
        self._update_final_report()

    def _update_final_report(self):
        parts: List[str] = []
        analyst_secs = [
            "market_report", "sentiment_report",
            "news_report", "fundamentals_report", "quant_report",
        ]
        if any(self.report_sections[s] for s in analyst_secs):
            parts.append("## Báo cáo nhóm phân tích")
            for sec in analyst_secs:
                val = self.report_sections.get(sec)
                if val:
                    parts.append(f"### {SECTION_TITLES[sec]}\n{val}")
        for key, label in [
            ("investment_plan",       "## ===== QUYẾT ĐỊNH NHÓM NGHIÊN CỨU ====="),
            ("trader_investment_plan","## ===== KẾ HOẠCH NHÓM GIAO DỊCH ====="),
            ("final_trade_decision",  "## ===== QUYẾT ĐỊNH CUỐI CÙNG ====="),
        ]:
            if self.report_sections.get(key):
                parts += ["---", label, str(self.report_sections[key])]
        self.final_report = "\n\n".join(parts) if parts else None


# ---------------------------------------------------------------------------
# Background analysis task
# ---------------------------------------------------------------------------

async def run_trading_analysis(
    session_id: str,
    ticker: str,
    analysis_date: str,
    config: dict,
    analysts: List[str],
    alpha_signal: Optional[Dict[str, Any]] = None,
):
    message_buffer = MessageBuffer()
    session_mgr.update(session_id, {
        "message_buffer":  message_buffer,
        "status":          "running",
        "current_step":    "Đang khởi tạo đồ thị phân tích",
        "progress_percent": 10,
        "cancel_requested": False,
    })
    _save_sessions_to_disk()

    try:
        if session_mgr.get_field(session_id, "cancel_requested"):
            raise asyncio.CancelledError("Analysis cancelled before start")

        graph = TradingAgentsGraph(debug=True, config=config, selected_analysts=list(analysts))

        analyst_map = {
            "market":       "Market Analyst",
            "social":       "Social Analyst",
            "news":         "News Analyst",
            "fundamentals": "Fundamentals Analyst",
        }
        for key in analysts:
            if key in analyst_map:
                message_buffer.update_agent_status(analyst_map[key], "in_progress")
        for key, name in analyst_map.items():
            if key not in analysts:
                message_buffer.agent_status[name] = "not_selected"

        max_debate_rounds = _to_positive_int(config.get("max_debate_rounds"), 1)
        max_risk_rounds   = _to_positive_int(config.get("max_risk_discuss_rounds"), 1)
        last_progress_persist = time.monotonic()

        # Map analyst key → report key
        report_by_analyst = {
            "market":       "market_report",
            "social":       "sentiment_report",
            "news":         "news_report",
            "fundamentals": "fundamentals_report",
        }

        def _on_graph_progress(chunk: Dict[str, Any]) -> None:
            nonlocal last_progress_persist
            if session_mgr.get_field(session_id, "cancel_requested"):
                raise asyncio.CancelledError("Analysis cancelled by user")

            progress_info = _derive_realtime_step(chunk, max_debate_rounds, max_risk_rounds)
            if progress_info is None:
                return

            step_text, step_percent = progress_info
            prev_pct = int(session_mgr.get_field(session_id, "progress_percent") or 0)
            session_mgr.update(session_id, {
                "current_step":    step_text,
                "progress_percent": max(prev_pct, step_percent),
            })

            # ── Analyst reports ──────────────────────────────────────────────
            # BUG FIX: chỉ set in_progress nếu chưa completed
            # Không reset analyst đã xong về in_progress khi chunk mới đến
            active_set = False
            for key in analysts:
                name = analyst_map.get(key)
                rkey = report_by_analyst.get(key)
                if not name or not rkey:
                    continue
                if chunk.get(rkey):
                    message_buffer.update_agent_status(name, "completed")
                elif message_buffer.agent_status.get(name) != "completed":
                    # Chỉ set in_progress cho analyst đầu tiên chưa xong
                    if not active_set:
                        message_buffer.update_agent_status(name, "in_progress")
                        active_set = True
                    # Các analyst sau để pending (không downgrade completed)

            # ── Investment debate ────────────────────────────────────────────
            invest_state = chunk.get("investment_debate_state")
            if isinstance(invest_state, dict):
                cr = str(invest_state.get("current_response", ""))
                jd = str(invest_state.get("judge_decision", ""))
                if cr.startswith("Bull Analyst"):
                    message_buffer.update_agent_status("Bull Researcher", "in_progress")
                elif cr.startswith("Bear Analyst"):
                    message_buffer.update_agent_status("Bull Researcher", "completed")
                    message_buffer.update_agent_status("Bear Researcher", "in_progress")
                if jd:
                    for a in ["Bull Researcher", "Bear Researcher"]:
                        message_buffer.update_agent_status(a, "completed")
                    message_buffer.update_agent_status("Research Manager", "in_progress")

            if chunk.get("investment_plan"):
                if message_buffer.agent_status.get("Research Manager") != "completed":
                    message_buffer.update_agent_status("Research Manager", "in_progress")

            if chunk.get("trader_investment_plan"):
                message_buffer.update_agent_status("Research Manager", "completed")
                message_buffer.update_agent_status("Trader", "completed")
            elif chunk.get("investment_plan"):
                message_buffer.update_agent_status("Trader", "in_progress")

            # ── Risk debate ──────────────────────────────────────────────────
            risk_state = chunk.get("risk_debate_state")
            if isinstance(risk_state, dict):
                ls = str(risk_state.get("latest_speaker", ""))
                jd = str(risk_state.get("judge_decision", ""))
                if ls == "AlphaGPT":
                    message_buffer.update_agent_status("AlphaGPT Analyst", "in_progress")
                elif ls == "Risky":
                    message_buffer.update_agent_status("AlphaGPT Analyst", "completed")
                    message_buffer.update_agent_status("Risky Analyst", "in_progress")
                elif ls == "Safe":
                    message_buffer.update_agent_status("Risky Analyst", "completed")
                    message_buffer.update_agent_status("Safe Analyst", "in_progress")
                elif ls == "Neutral":
                    message_buffer.update_agent_status("Safe Analyst", "completed")
                    message_buffer.update_agent_status("Neutral Analyst", "in_progress")
                elif ls == "Judge" or jd:
                    for a in ["AlphaGPT Analyst", "Risky Analyst", "Safe Analyst", "Neutral Analyst"]:
                        message_buffer.update_agent_status(a, "completed")
                    message_buffer.update_agent_status("Portfolio Manager", "in_progress")

            if chunk.get("final_trade_decision"):
                message_buffer.update_agent_status("Portfolio Manager", "completed")

            now = time.monotonic()
            if now - last_progress_persist >= 1.0:
                _save_sessions_to_disk()
                last_progress_persist = now

        def _run_graph():
            return graph.propagate(
                ticker,
                analysis_date,
                alphagpt_signal=alpha_signal,
                progress_callback=_on_graph_progress,
            )

        final_state, decision = await asyncio.to_thread(_run_graph)

        # ── Cập nhật report sections từ final_state ──────────────────────
        section_map = {
            "market_report":       "Market Analyst",
            "sentiment_report":    "Social Analyst",
            "news_report":         "News Analyst",
            "fundamentals_report": "Fundamentals Analyst",
            "quant_report":        "AlphaGPT Analyst",
        }
        for sec_key, analyst_name in section_map.items():
            if final_state.get(sec_key):
                message_buffer.update_report_section(sec_key, final_state[sec_key])
                message_buffer.update_agent_status(analyst_name, "completed")

        if final_state.get("investment_plan"):
            for a in ["Bull Researcher", "Bear Researcher"]:
                message_buffer.update_agent_status(a, "completed")
            message_buffer.update_report_section("investment_plan", final_state["investment_plan"])
            message_buffer.update_agent_status("Research Manager", "completed")

        if final_state.get("trader_investment_plan"):
            message_buffer.update_report_section("trader_investment_plan", final_state["trader_investment_plan"])
            message_buffer.update_agent_status("Trader", "completed")

        if final_state.get("final_trade_decision"):
            for a in ["Risky Analyst", "Safe Analyst", "Neutral Analyst"]:
                message_buffer.update_agent_status(a, "completed")
            message_buffer.update_report_section("final_trade_decision", str(final_state["final_trade_decision"]))
            message_buffer.update_agent_status("Portfolio Manager", "completed")

        # Đảm bảo mọi agent còn pending/in_progress đều completed
        for agent, status in list(message_buffer.agent_status.items()):
            if status not in {"not_selected", "completed"}:
                message_buffer.update_agent_status(agent, "completed")

        ta_decision_label = _extract_decision_label(decision)
        alpha_side = alpha_signal.get("side") if isinstance(alpha_signal, dict) else None
        fused_decision, fusion_note = _fuse_decision_with_alphagpt(ta_decision_label, alpha_side)

        session_mgr.update(session_id, {
            "status":               "completed",
            "current_step":         "✅ Hoàn thành phân tích",
            "progress_percent":     100,
            "decision":             decision,
            "decision_raw":         decision,
            "decision_fused":       fused_decision,
            "decision_fusion_note": fusion_note,
            "alpha_signal":         alpha_signal,
            "final_state":          final_state,
        })
        _save_sessions_to_disk()

    except asyncio.CancelledError as e:
        session_mgr.update(session_id, {
            "status":       "cancelled",
            "current_step": "🛑 Đã hủy phân tích theo yêu cầu",
            "error":        str(e),
            "error_details": None,
        })
        if not session_mgr.get_field(session_id, "progress_percent"):
            session_mgr.set_field(session_id, "progress_percent", 0)
        _save_sessions_to_disk()

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logger.error("Lỗi phân tích %s: %s", session_id, error_details)
        session_mgr.update(session_id, {
            "status":       "error",
            "current_step": "❌ Phân tích thất bại",
            "error":        str(e),
            "error_details": error_details,
        })
        if not session_mgr.get_field(session_id, "progress_percent"):
            session_mgr.set_field(session_id, "progress_percent", 0)
        mb = session_mgr.get_field(session_id, "message_buffer")
        if mb:
            mb.add_message("Error", f"Analysis failed: {str(e)}")
        _save_sessions_to_disk()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/market", response_class=HTMLResponse)
async def market_page(request: Request):
    return templates.TemplateResponse("market_new.html", {"request": request})


@app.post("/api/analyze", response_model=AnalysisResponse)
async def start_analysis(request: AnalysisRequest, background_tasks: BackgroundTasks):
    try:
        selected_analysts = request.analysts or ["market", "social", "news", "fundamentals"]
        session_id = f"{request.ticker}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

        trigger_if_needed()

        alpha_signal  = _load_alphagpt_signal(request.ticker)
        analysis_date = request.analysis_date or datetime.datetime.now().strftime("%Y-%m-%d")

        config = DEFAULT_CONFIG.copy()
        config["deep_think_llm"]  = request.deep_think_llm
        config["quick_think_llm"] = request.quick_think_llm
        config["max_debate_rounds"] = request.research_depth or request.max_debate_rounds
        config["data_vendors"]    = request.data_vendors

        session_mgr.set(session_id, {
            "ticker":           request.ticker,
            "analysis_date":    analysis_date,
            "analysts":         selected_analysts,
            "alpha_signal":     alpha_signal,
            "status":           "initializing",
            "current_step":     "Đang tạo phiên phân tích",
            "progress_percent": 5,
            "cancel_requested": False,
            "created_at":       datetime.datetime.now().isoformat(),
        })
        _save_sessions_to_disk()

        background_tasks.add_task(
            run_trading_analysis,
            session_id, request.ticker, analysis_date, config, selected_analysts, alpha_signal,
        )
        return AnalysisResponse(
            session_id=session_id, status="started",
            message=f"Analysis started for {request.ticker}",
        )

    except Exception as e:
        import traceback
        logger.error("Lỗi start_analysis: %s", traceback.format_exc())
        return JSONResponse(status_code=500, content={"error": str(e), "details": traceback.format_exc()})


@app.get("/api/status/{session_id}", response_model=AnalysisStatus)
async def get_analysis_status(session_id: str):
    session = session_mgr.get(session_id)
    if session is None:
        return JSONResponse(status_code=404, content={"error": "Session not found"})

    mb = session.get("message_buffer")
    if mb:
        return AnalysisStatus(
            session_id=session_id,
            status=session["status"],
            current_step=session.get("current_step"),
            progress_percent=session.get("progress_percent"),
            current_agent=mb.current_agent,
            agent_status=mb.agent_status,
            current_report=mb.current_report,
            final_report=mb.final_report,
            decision=session.get("decision"),
            decision_raw=session.get("decision_raw"),
            decision_fused=session.get("decision_fused"),
            decision_fusion_note=session.get("decision_fusion_note"),
            alpha_signal=session.get("alpha_signal"),
            messages=list(mb.messages),
            tool_calls=list(mb.tool_calls),
            error=session.get("error"),
            error_details=session.get("error_details"),
        )

    # Session load từ disk — dùng rebuild functions từ session_serialization
    final_state = session.get("final_state")
    rebuilt_current, rebuilt_final = rebuild_reports_from_final_state(final_state)
    loaded_agent_status = (
        session.get("agent_status")
        or rebuild_agent_status_from_final_state(final_state, str(session.get("status", "")))
    )

    return AnalysisStatus(
        session_id=session_id,
        status=session["status"],
        current_step=session.get("current_step"),
        progress_percent=session.get("progress_percent"),
        current_agent=session.get("current_agent"),
        agent_status=loaded_agent_status or {},
        current_report=session.get("current_report") or rebuilt_current,
        final_report=session.get("final_report") or rebuilt_final,
        decision=session.get("decision"),
        decision_raw=session.get("decision_raw"),
        decision_fused=session.get("decision_fused"),
        decision_fusion_note=session.get("decision_fusion_note"),
        alpha_signal=session.get("alpha_signal"),
        messages=session.get("messages") or [],
        tool_calls=session.get("tool_calls") or [],
        error=session.get("error"),
        error_details=session.get("error_details"),
    )


@app.get("/api/sessions")
async def list_sessions():
    return {
        "sessions": [
            {
                "session_id":    sid,
                "ticker":        data.get("ticker"),
                "analysis_date": data.get("analysis_date"),
                "status":        data.get("status"),
                "created_at":    data.get("created_at"),
            }
            for sid, data in session_mgr.all_items()
        ]
    }


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    if session_mgr.delete(session_id):
        _save_sessions_to_disk()
        return {"message": "Session deleted successfully"}
    return JSONResponse(status_code=404, content={"error": "Session not found"})


@app.post("/api/sessions/{session_id}/cancel")
async def cancel_session(session_id: str):
    session = session_mgr.get(session_id)
    if session is None:
        return JSONResponse(status_code=404, content={"error": "Session not found"})
    status = str(session.get("status", ""))
    if status in {"completed", "error", "cancelled"}:
        return {"message": f"Session is already {status}", "status": status}
    session_mgr.update(session_id, {
        "cancel_requested": True,
        "current_step":     "🛑 Đang hủy phân tích...",
    })
    _save_sessions_to_disk()
    return {"message": "Cancellation requested", "status": session.get("status")}


@app.get("/health")
async def health_check():
    return {
        "status":         "healthy",
        "timestamp":      datetime.datetime.now().isoformat(),
        "alpha_pipeline": get_status(),
    }


@app.get("/api/alpha/status")
async def alpha_status():
    return get_status()


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def on_startup():
    _load_sessions_from_disk()
    try:
        init_alpha_manager()
        logger.info("Alpha manager đã được khởi tạo.")
    except Exception as exc:
        logger.error("Không thể khởi tạo alpha manager: %s", exc, exc_info=True)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.datetime.now().isoformat(),
        "alpha_pipeline": get_status(),
    }


@app.get("/api/alpha/status")
async def alpha_status():
    """Endpoint riêng để xem trạng thái alpha pipeline."""
    return get_status()


# ---------------------------------------------------------------------------
# Startup / Shutdown
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def on_startup():
    _load_sessions_from_disk()
    # Khởi tạo alpha manager và trigger pipeline nếu cần
    try:
        init_alpha_manager()
        logger.info("Alpha manager đã được khởi tạo.")
    except Exception as exc:
        logger.error("Không thể khởi tạo alpha manager: %s", exc, exc_info=True)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
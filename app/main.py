from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List, Tuple
import datetime
from pathlib import Path
import asyncio
from collections import deque
import csv
import os
import time
from threading import Lock

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG
from dotenv import load_dotenv
from app.routes.market import router as market_router
from app.services.session_serialization import build_persistable_session
from app.storage.session_store import SQLiteSessionStore

# Load environment variables
load_dotenv()

app = FastAPI(
    title="TradingAgents API",
    description="Multi-Agents LLM Financial Trading Framework",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(market_router)

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Templates
templates = Jinja2Templates(directory="app/templates")

# Global state for analysis sessions
analysis_sessions: Dict[str, Dict[str, Any]] = {}
analysis_sessions_lock = Lock()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SESSION_DB_PATH = PROJECT_ROOT / "app" / "data" / "sessions.db"
LEGACY_SESSION_JSON_PATH = PROJECT_ROOT / "app" / "data" / "sessions.json"
DEFAULT_ALPHAGPT_SIGNAL_CSV = (
    PROJECT_ROOT.parent / "AlphaGPT" / "alpha-gpt" / "outputs" / "signal_scores_auto_best.csv"
)
session_store = SQLiteSessionStore(SESSION_DB_PATH)


def _save_sessions_to_disk() -> None:
    with analysis_sessions_lock:
        persistable_sessions = {
            session_id: build_persistable_session(session_data)
            for session_id, session_data in analysis_sessions.items()
        }
    session_store.save_all(persistable_sessions)


def _load_sessions_from_disk() -> None:
    session_store.migrate_from_json_file(LEGACY_SESSION_JSON_PATH)
    try:
        sessions = session_store.load_all()
        for session_id, session_data in sessions.items():
            status = session_data.get("status")
            if status in {"initializing", "running"}:
                session_data["status"] = "error"
                session_data["error"] = "Session interrupted because server restarted"
            analysis_sessions[session_id] = session_data
    except Exception as exc:
        print(f"Warning: unable to load sessions from sqlite: {exc}")


def _get_alphagpt_signal_csv_path() -> Path:
    configured_path = os.getenv("ALPHAGPT_SIGNAL_CSV", "").strip()
    if configured_path:
        path = Path(configured_path)
        if not path.is_absolute():
            path = (PROJECT_ROOT / path).resolve()
        return path
    return DEFAULT_ALPHAGPT_SIGNAL_CSV


def _load_alphagpt_signal_for_ticker(ticker: str) -> Dict[str, Any]:
    ticker_symbol = ticker.strip().upper()
    csv_path = _get_alphagpt_signal_csv_path()
    signal: Dict[str, Any] = {
        "enabled": False,
        "ticker": ticker_symbol,
        "source": str(csv_path),
        "side": "neutral",
        "score": None,
        "rank": None,
    }

    if not csv_path.exists():
        signal["error"] = "AlphaGPT signal CSV not found"
        return signal

    try:
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                symbol = str(row.get("symbol", "")).strip().upper()
                if symbol != ticker_symbol:
                    continue

                side = str(row.get("side", "neutral")).strip().lower() or "neutral"
                if side not in {"long", "short", "neutral"}:
                    side = "neutral"

                score_value = row.get("score")
                rank_value = row.get("rank")
                signal.update(
                    {
                        "enabled": True,
                        "symbol": symbol,
                        "side": side,
                        "score": float(score_value) if score_value not in (None, "") else None,
                        "rank": int(float(rank_value)) if rank_value not in (None, "") else None,
                    }
                )
                return signal
    except Exception as exc:
        signal["error"] = f"Failed to read AlphaGPT signal CSV: {exc}"
        return signal

    signal["error"] = "Ticker not found in AlphaGPT signal CSV"
    return signal


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


def _rebuild_agent_status_from_final_state(final_state: Any, session_status: str) -> Dict[str, str]:
    if session_status == "completed":
        return {agent: "completed" for agent in MessageBuffer().agent_status}

    if not isinstance(final_state, dict):
        return {}

    status_map = {agent: "pending" for agent in MessageBuffer().agent_status}
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


def _extract_tool_call_names(message: Any) -> List[str]:
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


def _to_positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
        return parsed if parsed > 0 else default
    except (TypeError, ValueError):
        return default


def _derive_realtime_step(
    chunk: Dict[str, Any],
    max_debate_rounds: int = 1,
    max_risk_rounds: int = 1,
) -> Optional[Tuple[str, int]]:
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
    decision: Optional[Any] = None  # Can be string or dict
    decision_raw: Optional[Any] = None
    decision_fused: Optional[str] = None
    decision_fusion_note: Optional[str] = None
    alpha_signal: Optional[Dict[str, Any]] = None
    messages: List[Dict[str, Any]] = []
    tool_calls: List[Dict[str, Any]] = []
    error: Optional[str] = None
    error_details: Optional[str] = None


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
            "investment_plan": None,
            "trader_investment_plan": None,
            "final_trade_decision": None,
        }

    def add_message(self, message_type, content):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.messages.append({
            "timestamp": timestamp,
            "type": message_type,
            "content": content
        })

    def add_tool_call(self, tool_name, args):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.tool_calls.append({
            "timestamp": timestamp,
            "tool": tool_name,
            "args": args
        })

    def update_agent_status(self, agent, status):
        if agent in self.agent_status:
            self.agent_status[agent] = status
            self.current_agent = agent

    def update_report_section(self, section_name, content):
        if section_name in self.report_sections:
            self.report_sections[section_name] = content
            self._update_current_report()

    def _update_current_report(self):
        latest_section = None
        latest_content = None

        for section, content in self.report_sections.items():
            if content is not None:
                latest_section = section
                latest_content = content

        if latest_section and latest_content:
            section_titles = {
                "market_report": "Phân tích thị trường",
                "sentiment_report": "Phân tích tâm lý xã hội",
                "news_report": "Phân tích tin tức",
                "fundamentals_report": "Phân tích cơ bản",
                "investment_plan": "Quyết định nhóm nghiên cứu",
                "trader_investment_plan": "Kế hoạch nhóm giao dịch",
                "final_trade_decision": "Quyết định cuối cùng",
            }
            self.current_report = f"### {section_titles[latest_section]}\n{latest_content}"

        self._update_final_report()

    def _update_final_report(self):
        report_parts = []

        if any(self.report_sections[section] for section in [
            "market_report", "sentiment_report", "news_report", "fundamentals_report"
        ]):
            report_parts.append("## Báo cáo nhóm phân tích")
            if self.report_sections["market_report"]:
                report_parts.append(f"### Phân tích thị trường\n{self.report_sections['market_report']}")
            if self.report_sections["sentiment_report"]:
                report_parts.append(f"### Phân tích tâm lý xã hội\n{self.report_sections['sentiment_report']}")
            if self.report_sections["news_report"]:
                report_parts.append(f"### Phân tích tin tức\n{self.report_sections['news_report']}")
            if self.report_sections["fundamentals_report"]:
                report_parts.append(f"### Phân tích cơ bản\n{self.report_sections['fundamentals_report']}")

        if self.report_sections["investment_plan"]:
            report_parts.append("---")
            report_parts.append("## ===== QUYẾT ĐỊNH NHÓM NGHIÊN CỨU =====")
            report_parts.append(f"{self.report_sections['investment_plan']}")

        if self.report_sections["trader_investment_plan"]:
            report_parts.append("---")
            report_parts.append("## ===== KẾ HOẠCH NHÓM GIAO DỊCH =====")
            report_parts.append(f"{self.report_sections['trader_investment_plan']}")

        if self.report_sections["final_trade_decision"]:
            report_parts.append("---")
            report_parts.append("## ===== QUYẾT ĐỊNH CUỐI CÙNG =====")
            report_parts.append(f"{self.report_sections['final_trade_decision']}")

        self.final_report = "\n\n".join(report_parts) if report_parts else None


async def run_trading_analysis(
    session_id: str,
    ticker: str,
    analysis_date: str,
    config: dict,
    analysts: List[str],
    alpha_signal: Optional[Dict[str, Any]] = None,
):
    """Run trading analysis in background"""
    message_buffer = MessageBuffer()
    analysis_sessions[session_id]["message_buffer"] = message_buffer
    analysis_sessions[session_id]["status"] = "running"
    analysis_sessions[session_id]["current_step"] = "Đang khởi tạo đồ thị phân tích"
    analysis_sessions[session_id]["progress_percent"] = 10
    analysis_sessions[session_id]["cancel_requested"] = False
    _save_sessions_to_disk()
    
    try:
        if analysis_sessions[session_id].get("cancel_requested"):
            raise asyncio.CancelledError("Analysis cancelled before start")

        selected_analysts = list(analysts)

        # Initialize trading graph
        graph = TradingAgentsGraph(debug=True, config=config, selected_analysts=selected_analysts)
        
        # Mark all selected analysts as in progress
        analyst_map = {
            "market": "Market Analyst",
            "social": "Social Analyst", 
            "news": "News Analyst",
            "fundamentals": "Fundamentals Analyst"
        }
        for analyst_key in selected_analysts:
            if analyst_key in analyst_map:
                message_buffer.update_agent_status(analyst_map[analyst_key], "in_progress")

        for analyst_key, agent_name in analyst_map.items():
            if analyst_key not in selected_analysts:
                message_buffer.agent_status[agent_name] = "not_selected"

        max_debate_rounds = _to_positive_int(config.get("max_debate_rounds"), 1)
        max_risk_rounds = _to_positive_int(config.get("max_risk_discuss_rounds"), 1)
        
        last_progress_persist = time.monotonic()

        def _on_graph_progress(chunk: Dict[str, Any]) -> None:
            nonlocal last_progress_persist

            if analysis_sessions[session_id].get("cancel_requested"):
                raise asyncio.CancelledError("Analysis cancelled by user")

            progress_info = _derive_realtime_step(
                chunk,
                max_debate_rounds=max_debate_rounds,
                max_risk_rounds=max_risk_rounds,
            )
            if progress_info is None:
                return

            step_text, step_percent = progress_info
            analysis_sessions[session_id]["current_step"] = step_text
            previous_percent = int(analysis_sessions[session_id].get("progress_percent") or 0)
            analysis_sessions[session_id]["progress_percent"] = max(previous_percent, step_percent)

            report_by_analyst = {
                "market": "market_report",
                "social": "sentiment_report",
                "news": "news_report",
                "fundamentals": "fundamentals_report",
            }

            first_incomplete_marked = False
            for analyst_key in selected_analysts:
                agent_name = analyst_map.get(analyst_key)
                report_key = report_by_analyst.get(analyst_key)
                if not agent_name or not report_key:
                    continue

                if chunk.get(report_key):
                    message_buffer.update_agent_status(agent_name, "completed")
                    continue

                if not first_incomplete_marked:
                    message_buffer.update_agent_status(agent_name, "in_progress")
                    first_incomplete_marked = True
                else:
                    message_buffer.update_agent_status(agent_name, "pending")

            investment_state = chunk.get("investment_debate_state")
            if isinstance(investment_state, dict):
                current_response = str(investment_state.get("current_response", ""))
                judge_decision = str(investment_state.get("judge_decision", ""))
                if current_response.startswith("Bull Analyst"):
                    message_buffer.update_agent_status("Bull Researcher", "in_progress")
                elif current_response.startswith("Bear Analyst"):
                    message_buffer.update_agent_status("Bull Researcher", "completed")
                    message_buffer.update_agent_status("Bear Researcher", "in_progress")
                if judge_decision:
                    message_buffer.update_agent_status("Bull Researcher", "completed")
                    message_buffer.update_agent_status("Bear Researcher", "completed")
                    message_buffer.update_agent_status("Research Manager", "completed")

            if chunk.get("investment_plan") and message_buffer.agent_status.get("Research Manager") != "completed":
                message_buffer.update_agent_status("Research Manager", "in_progress")

            if chunk.get("trader_investment_plan"):
                message_buffer.update_agent_status("Trader", "completed")
            elif chunk.get("investment_plan"):
                message_buffer.update_agent_status("Trader", "in_progress")

            risk_state = chunk.get("risk_debate_state")
            if isinstance(risk_state, dict):
                latest_speaker = str(risk_state.get("latest_speaker", ""))
                if latest_speaker == "Risky":
                    message_buffer.update_agent_status("AlphaGPT Analyst", "completed")
                    message_buffer.update_agent_status("Risky Analyst", "in_progress")
                elif latest_speaker == "Safe":
                    message_buffer.update_agent_status("Risky Analyst", "completed")
                    message_buffer.update_agent_status("Safe Analyst", "in_progress")
                elif latest_speaker == "Neutral":
                    message_buffer.update_agent_status("Safe Analyst", "completed")
                    message_buffer.update_agent_status("Neutral Analyst", "in_progress")
                elif latest_speaker == "Judge":
                    message_buffer.update_agent_status("Risky Analyst", "completed")
                    message_buffer.update_agent_status("Safe Analyst", "completed")
                    message_buffer.update_agent_status("Neutral Analyst", "completed")
                    message_buffer.update_agent_status("Portfolio Manager", "in_progress")

            if chunk.get("final_trade_decision"):
                message_buffer.update_agent_status("Portfolio Manager", "completed")

            now = time.monotonic()
            if now - last_progress_persist >= 1.0:
                _save_sessions_to_disk()
                last_progress_persist = now

        def _run_graph() -> Tuple[Dict[str, Any], Any]:
            return graph.propagate(
                ticker,
                analysis_date,
                alphagpt_signal=alpha_signal,
                progress_callback=_on_graph_progress,
            )

        # Run graph in a worker thread to keep API status polling responsive.
        final_state, decision = await asyncio.to_thread(_run_graph)

        alpha_note = "AlphaGPT signal routed through AlphaGPT Analyst node in decision layer"
        if isinstance(alpha_signal, dict) and not alpha_signal.get("enabled", False):
            alpha_note = "AlphaGPT Analyst active but no directional signal available"
        
        # Process final state to extract reports
        section_map = {
            "market_report": "Market Analyst",
            "sentiment_report": "Social Analyst",
            "news_report": "News Analyst",
            "fundamentals_report": "Fundamentals Analyst",
        }
        
        # Update analyst reports
        for section_key, analyst_name in section_map.items():
            if section_key in final_state and final_state[section_key]:
                message_buffer.update_report_section(section_key, final_state[section_key])
                message_buffer.update_agent_status(analyst_name, "completed")
        
        # Update research team
        if "investment_plan" in final_state and final_state["investment_plan"]:
            message_buffer.update_agent_status("Research Manager", "in_progress")
            message_buffer.update_agent_status("Bull Researcher", "completed")
            message_buffer.update_agent_status("Bear Researcher", "completed")
            message_buffer.update_report_section("investment_plan", final_state["investment_plan"])
            message_buffer.update_agent_status("Research Manager", "completed")
        
        # Update trader
        if "trader_investment_plan" in final_state and final_state["trader_investment_plan"]:
            message_buffer.update_agent_status("Trader", "in_progress")
            message_buffer.update_report_section("trader_investment_plan", final_state["trader_investment_plan"])
            message_buffer.update_agent_status("Trader", "completed")

        # AlphaGPT decision-layer input
        message_buffer.update_agent_status("AlphaGPT Analyst", "in_progress")
        message_buffer.update_agent_status("AlphaGPT Analyst", "completed")
        
        # Update risk management
        if "final_trade_decision" in final_state and final_state["final_trade_decision"]:
            message_buffer.update_agent_status("Portfolio Manager", "in_progress")
            message_buffer.update_agent_status("Risky Analyst", "completed")
            message_buffer.update_agent_status("Safe Analyst", "completed")
            message_buffer.update_agent_status("Neutral Analyst", "completed")
            message_buffer.update_report_section(
                "final_trade_decision",
                f"{final_state['final_trade_decision']}"
            )
            message_buffer.update_agent_status("Portfolio Manager", "completed")
        
        # Finalize remaining active agents; keep unselected analysts untouched.
        for agent, status in list(message_buffer.agent_status.items()):
            if status == "not_selected":
                continue
            if status in {"pending", "in_progress"}:
                message_buffer.update_agent_status(agent, "completed")

        # Store final results
        analysis_sessions[session_id]["status"] = "completed"
        analysis_sessions[session_id]["current_step"] = "✅ Hoàn thành phân tích"
        analysis_sessions[session_id]["progress_percent"] = 100
        analysis_sessions[session_id]["decision"] = decision
        analysis_sessions[session_id]["decision_raw"] = decision
        analysis_sessions[session_id]["decision_fused"] = _extract_decision_label(decision)
        analysis_sessions[session_id]["decision_fusion_note"] = alpha_note
        analysis_sessions[session_id]["alpha_signal"] = alpha_signal
        analysis_sessions[session_id]["final_state"] = final_state
        _save_sessions_to_disk()

    except asyncio.CancelledError as e:
        analysis_sessions[session_id]["status"] = "cancelled"
        analysis_sessions[session_id]["current_step"] = "🛑 Đã hủy phân tích theo yêu cầu"
        analysis_sessions[session_id]["error"] = str(e)
        analysis_sessions[session_id]["error_details"] = None
        if analysis_sessions[session_id].get("progress_percent") is None:
            analysis_sessions[session_id]["progress_percent"] = 0
        _save_sessions_to_disk()
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        analysis_sessions[session_id]["status"] = "error"
        analysis_sessions[session_id]["current_step"] = "❌ Phân tích thất bại"
        if analysis_sessions[session_id].get("progress_percent") is None:
            analysis_sessions[session_id]["progress_percent"] = 0
        analysis_sessions[session_id]["error"] = str(e)
        analysis_sessions[session_id]["error_details"] = error_details
        print(f"Error in analysis {session_id}: {error_details}")  # Log to console
        
        # Update message buffer with error
        if "message_buffer" in analysis_sessions[session_id]:
            message_buffer = analysis_sessions[session_id]["message_buffer"]
            message_buffer.add_message("Error", f"Analysis failed: {str(e)}")
        _save_sessions_to_disk()


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/market", response_class=HTMLResponse)
async def market_page(request: Request):
    """Stock market page"""
    return templates.TemplateResponse("market_new.html", {"request": request})


@app.post("/api/analyze", response_model=AnalysisResponse)
async def start_analysis(request: AnalysisRequest, background_tasks: BackgroundTasks):
    """Start a new trading analysis"""
    try:
        selected_analysts: List[str] = request.analysts or ["market", "social", "news", "fundamentals"]

        # Generate session ID
        session_id = f"{request.ticker}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        alpha_signal = _load_alphagpt_signal_for_ticker(request.ticker)
        
        # Set default analysis date if not provided
        analysis_date = request.analysis_date or datetime.datetime.now().strftime("%Y-%m-%d")
        
        # Create config
        config = DEFAULT_CONFIG.copy()
        config["deep_think_llm"] = request.deep_think_llm
        config["quick_think_llm"] = request.quick_think_llm
        config["max_debate_rounds"] = request.research_depth or request.max_debate_rounds
        config["data_vendors"] = request.data_vendors
        
        # Initialize session
        analysis_sessions[session_id] = {
            "ticker": request.ticker,
            "analysis_date": analysis_date,
            "analysts": selected_analysts,
            "alpha_signal": alpha_signal,
            "status": "initializing",
            "current_step": "Đang tạo phiên phân tích",
            "progress_percent": 5,
            "cancel_requested": False,
            "created_at": datetime.datetime.now().isoformat(),
        }
        _save_sessions_to_disk()
        
        # Run analysis in background
        background_tasks.add_task(
            run_trading_analysis,
            session_id,
            request.ticker,
            analysis_date,
            config,
            selected_analysts,
            alpha_signal,
        )
        
        return AnalysisResponse(
            session_id=session_id,
            status="started",
            message=f"Analysis started for {request.ticker}"
        )
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error starting analysis: {error_details}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "details": error_details}
        )


@app.get("/api/status/{session_id}", response_model=AnalysisStatus)
async def get_analysis_status(session_id: str):
    """Get status of an analysis session"""
    if session_id not in analysis_sessions:
        return JSONResponse(
            status_code=404,
            content={"error": "Session not found"}
        )
    
    session = analysis_sessions[session_id]
    message_buffer = session.get("message_buffer")
    
    if message_buffer:
        return AnalysisStatus(
            session_id=session_id,
            status=session["status"],
            current_step=session.get("current_step"),
            progress_percent=session.get("progress_percent"),
            current_agent=message_buffer.current_agent,
            agent_status=message_buffer.agent_status,
            current_report=message_buffer.current_report,
            final_report=message_buffer.final_report,
            decision=session.get("decision"),
            decision_raw=session.get("decision_raw"),
            decision_fused=session.get("decision_fused"),
            decision_fusion_note=session.get("decision_fusion_note"),
            alpha_signal=session.get("alpha_signal"),
            messages=list(message_buffer.messages),
            tool_calls=list(message_buffer.tool_calls),
            error=session.get("error"),
            error_details=session.get("error_details"),
        )
    else:
        loaded_agent_status = session.get("agent_status")
        if not isinstance(loaded_agent_status, dict):
            loaded_agent_status = {}

        loaded_messages = session.get("messages")
        if not isinstance(loaded_messages, list):
            loaded_messages = []

        loaded_tool_calls = session.get("tool_calls")
        if not isinstance(loaded_tool_calls, list):
            loaded_tool_calls = []

        loaded_current_report = session.get("current_report")
        loaded_final_report = session.get("final_report")
        final_state = session.get("final_state")

        rebuilt_current, rebuilt_final = _rebuild_reports_from_final_state(final_state)
        if rebuilt_current:
            loaded_current_report = rebuilt_current
        if rebuilt_final:
            loaded_final_report = rebuilt_final

        if not loaded_agent_status:
            loaded_agent_status = _rebuild_agent_status_from_final_state(
                final_state,
                str(session.get("status", "")),
            )

        return AnalysisStatus(
            session_id=session_id,
            status=session["status"],
            current_step=session.get("current_step"),
            progress_percent=session.get("progress_percent"),
            current_agent=session.get("current_agent"),
            agent_status=loaded_agent_status,
            current_report=loaded_current_report,
            final_report=loaded_final_report,
            decision_raw=session.get("decision_raw"),
            decision=session.get("decision"),
            decision_fused=session.get("decision_fused"),
            decision_fusion_note=session.get("decision_fusion_note"),
            alpha_signal=session.get("alpha_signal"),
            messages=loaded_messages,
            tool_calls=loaded_tool_calls,
            error=session.get("error"),
            error_details=session.get("error_details"),
        )


@app.get("/api/sessions")
async def list_sessions():
    """List all analysis sessions"""
    sessions = []
    for session_id, session_data in analysis_sessions.items():
        sessions.append({
            "session_id": session_id,
            "ticker": session_data.get("ticker"),
            "analysis_date": session_data.get("analysis_date"),
            "status": session_data.get("status"),
            "created_at": session_data.get("created_at"),
        })
    return {"sessions": sessions}


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete an analysis session"""
    if session_id in analysis_sessions:
        del analysis_sessions[session_id]
        _save_sessions_to_disk()
        return {"message": "Session deleted successfully"}
    return JSONResponse(
        status_code=404,
        content={"error": "Session not found"}
    )


@app.post("/api/sessions/{session_id}/cancel")
async def cancel_session(session_id: str):
    """Request cancellation of a running session."""
    if session_id not in analysis_sessions:
        return JSONResponse(
            status_code=404,
            content={"error": "Session not found"}
        )

    session = analysis_sessions[session_id]
    status = str(session.get("status", ""))
    if status in {"completed", "error", "cancelled"}:
        return {
            "message": f"Session is already {status}",
            "status": status,
        }

    session["cancel_requested"] = True
    session["current_step"] = "🛑 Đang hủy phân tích..."
    _save_sessions_to_disk()
    return {"message": "Cancellation requested", "status": session.get("status")}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.datetime.now().isoformat()}


@app.on_event("startup")
async def load_session_store_on_startup():
    _load_sessions_from_disk()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

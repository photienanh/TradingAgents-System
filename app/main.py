"""
app/main.py
"""

import datetime
import asyncio
import os
import time
import logging
from pathlib import Path
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
    _normalize_section,
    SECTION_TITLES,
    DEFAULT_AGENT_NAMES,
)
from app.services.session_manager import SessionManager
from app.storage.session_store import SQLiteSessionStore
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
# Generic helpers
# ---------------------------------------------------------------------------

def _load_alphagpt_signal(ticker: str) -> Dict[str, Any]:
    return get_signal_for_ticker(ticker)


def _init_vnstock_auth() -> None:
    api_key = (os.getenv("VNSTOCK_API_KEY") or "").strip()
    if not api_key:
        logger.warning("VNSTOCK_API_KEY chưa được thiết lập; bỏ qua đăng nhập vnstock.")
        return
    try:
        from vnstock import register_user
        register_user(api_key)
        logger.info("Đăng nhập vnstock thành công.")
    except Exception as exc:
        logger.warning("Đăng nhập/import vnstock thất bại: %s", exc)


def _extract_decision_label(raw_decision: Any) -> str:
    text = str(raw_decision or "").upper()
    if "BUY"  in text: return "BUY"
    if "SELL" in text: return "SELL"
    return "HOLD"


def _fuse_decision_with_alphagpt(
    ta_decision: str, alpha_signal: Optional[Dict[str, Any]],
    ic_threshold: float = 0.10,
    long_score_threshold: float = 0.5,
    short_score_threshold: float = -1.0,
) -> Tuple[str, str]:
    decision = ta_decision if ta_decision in {"BUY", "SELL", "HOLD"} else "HOLD"
    if not isinstance(alpha_signal, dict) or not alpha_signal.get("enabled"):
        return decision, "AlphaGPT signal unavailable → TA decision kept"
    side   = (alpha_signal.get("side") or "neutral").strip().lower()
    ic_oos = float(alpha_signal.get("ic_oos") or 0.0)
    score  = float(alpha_signal.get("score") or 0.0)
    if side not in ("long", "short"):
        return decision, f"AlphaGPT neutral (side={side}) → TA decision kept"
    if ic_oos < ic_threshold:
        return decision, (
            f"AlphaGPT weak: ic_oos={ic_oos:.4f} < {ic_threshold} → TA decision kept ({decision})"
        )
    if side == "long":
        if score <= long_score_threshold:
            return decision, (
                f"AlphaGPT long weak: z={score:.3f} <= {long_score_threshold} → TA decision kept ({decision})"
            )
        if decision == "HOLD":  return "BUY",  f"AlphaGPT long (ic_oos={ic_oos:.4f}, z={score:.3f}) upgraded HOLD → BUY"
        if decision == "SELL":  return "HOLD", f"Conflict: TA=SELL vs AlphaGPT long (ic_oos={ic_oos:.4f}) → HOLD"
        return "BUY", f"TA=BUY aligned with AlphaGPT long (ic_oos={ic_oos:.4f}, z={score:.3f})"
    # short
    if score >= short_score_threshold:
        return decision, (
            f"AlphaGPT short weak: z={score:.3f} >= {short_score_threshold} → TA decision kept ({decision})"
        )
    if decision == "HOLD":  return "SELL", f"AlphaGPT short (ic_oos={ic_oos:.4f}, z={score:.3f}) downgraded HOLD → SELL"
    if decision == "BUY":   return "HOLD", f"Conflict: TA=BUY vs AlphaGPT short (ic_oos={ic_oos:.4f}) → HOLD"
    return "SELL", f"TA=SELL aligned with AlphaGPT short (ic_oos={ic_oos:.4f}, z={score:.3f})"


def _to_positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
        return parsed if parsed > 0 else default
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Chunk reading helpers (all work on flat AgentState from stream_mode='values')
# ---------------------------------------------------------------------------

def _get_last_message(chunk: Dict[str, Any]) -> Any:
    msgs = chunk.get("messages")
    if not isinstance(msgs, list) or not msgs:
        return None
    return msgs[-1]


def _last_msg_tool_calls(chunk: Dict[str, Any]) -> List[str]:
    """Return list of tool call names from the last message."""
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


def _last_msg_is_clear(chunk: Dict[str, Any]) -> bool:
    """True when Msg Clear node just fired: last message is HumanMessage('Continue')."""
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


# ---------------------------------------------------------------------------
# Analyst metadata (single source of truth)
# ---------------------------------------------------------------------------

_ANALYST_ORDER = ["market", "social", "news", "fundamentals"]

_ANALYST_DISPLAY: Dict[str, str] = {
    "market":       "Market Analyst",
    "social":       "Social Analyst",
    "news":         "News Analyst",
    "fundamentals": "Fundamentals Analyst",
}

_REPORT_KEY: Dict[str, str] = {
    "market":       "market_report",
    "social":       "sentiment_report",
    "news":         "news_report",
    "fundamentals": "fundamentals_report",
}

# Which analyst owns each tool
_TOOL_ANALYST: Dict[str, str] = {
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

# Human-readable step label for each tool call
_TOOL_STEP_LABEL: Dict[str, str] = {
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

_ANALYST_RUNNING_LABEL: Dict[str, str] = {
    "market":       "Market Analyst đang phân tích...",
    "social":       "Social Analyst đang phân tích...",
    "news":         "News Analyst đang phân tích...",
    "fundamentals": "Fundamentals Analyst đang phân tích...",
}


# ---------------------------------------------------------------------------
# Progress derivation
# ---------------------------------------------------------------------------

def _derive_realtime_step(
    chunk: Dict[str, Any],
    max_debate_rounds: int = 1,
    max_risk_rounds: int = 1,
    selected_analysts: Optional[List[str]] = None,
) -> Optional[Tuple[str, int]]:
    """
    Returns (step_label, progress_pct) or None.
    chunk is always a flat AgentState from stream_mode='values'.

    Design principle: ALWAYS show what is CURRENTLY HAPPENING (đang...),
    never show 'đã hoàn thành' as the primary progress text — that belongs
    in the agent status panel, not the step label.

    Progress bands:
      10–15  : init
      15–72  : analyst team (evenly split)
      72–77  : AlphaGPT Analyst
      77–88  : researcher debate
      88–90  : trader
      90–99  : risk debate
      100    : done
    """
    import math as _math

    max_debate_rounds = _to_positive_int(max_debate_rounds, 1)
    max_risk_rounds   = _to_positive_int(max_risk_rounds,   1)

    active = [k for k in _ANALYST_ORDER if k in (selected_analysts or _ANALYST_ORDER)]
    n = max(1, len(active))
    A_START, A_END = 15, 72

    # Count completed analyst reports
    done_keys = [k for k in active if chunk.get(_REPORT_KEY[k])]
    n_done    = len(done_keys)

    # ── A. Tool call in progress (highest priority) ──────────────────────
    # Shows exact data-fetch action and overrides any state-based label.
    tool_names = _last_msg_tool_calls(chunk)
    if tool_names:
        first_tool  = tool_names[0]
        analyst_key = _TOOL_ANALYST.get(first_tool)
        label = _TOOL_STEP_LABEL.get(first_tool, "Đang thu thập dữ liệu...")
        slot_w = (A_END - A_START) / n
        if analyst_key and analyst_key in active:
            slot = active.index(analyst_key)
            pct  = int(A_START + slot * slot_w + slot_w / 2)
        else:
            pct = int(A_START + (A_END - A_START) * n_done / n)
        return (label, pct)

    # ── B. Final decision done ───────────────────────────────────────────
    if chunk.get("final_trade_decision"):
        return ("Hoàn thành phán quyết cuối cùng", 100)

    # ── C. All analysts done → downstream stages ─────────────────────────
    if n_done == n:
        slot_w = (A_END - A_START) / n

        # Risk debate
        if chunk.get("trader_investment_plan"):
            risk_state = chunk.get("risk_debate_state")
            if isinstance(risk_state, dict):
                ls = str(risk_state.get("latest_speaker", ""))
                jd = str(risk_state.get("judge_decision", ""))
                if jd or ls == "Judge":
                    return ("Portfolio Manager đang chốt quyết định cuối...", 99)
                max_t   = max(1, 3 * max_risk_rounds)
                count   = _to_positive_int(risk_state.get("count"), 1)
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

        # Trader running
        if chunk.get("investment_plan"):
            return ("Trader đang lập kế hoạch giao dịch...", 89)

        # Researcher debate
        invest_state = chunk.get("investment_debate_state")
        if isinstance(invest_state, dict):
            jd    = str(invest_state.get("judge_decision", ""))
            cr    = str(invest_state.get("current_response", ""))
            count = _to_positive_int(invest_state.get("count"), 0)
            if jd:
                return ("Research Manager đang chốt phán quyết...", 87)
            if count > 0:
                # Fix issue 3: round = ceil(count / 2), max_rounds = max_debate_rounds
                # count=1 → Bull round 1, count=2 → Bear round 1
                # count=3 → Bull round 2, count=4 → Bear round 2, etc.
                round_num = _math.ceil(count / 2)
                round_num = min(round_num, max_debate_rounds)
                dpct = 78 + int((count / (2 * max_debate_rounds)) * 8)
                dpct = min(dpct, 86)
                is_bull = (count % 2 == 1)
                speaker = "Bull Researcher" if is_bull else "Bear Researcher"
                return (
                    f"{speaker} đang tranh luận (vòng {round_num}/{max_debate_rounds})...",
                    dpct,
                )

        # AlphaGPT done → Bull about to start
        if chunk.get("quant_report"):
            return ("Bull Researcher đang bắt đầu tranh luận...", 78)

        # AlphaGPT running
        return ("AlphaGPT Analyst đang xử lý...", 74)

    # ── D. Analyst team in progress ──────────────────────────────────────
    # Fix issue 1 & 2: always show the NEXT/CURRENT analyst (đang...),
    # not the last completed one. With stream_mode=values, once market_report
    # is non-empty it stays non-empty in all subsequent chunks — so showing
    # 'Market Analyst done' again is wrong when Social is already running.
    slot_w = (A_END - A_START) / n

    if n_done < n:
        # The currently active analyst = first one with empty report
        current_key = next((k for k in active if not chunk.get(_REPORT_KEY[k])), None)
        if current_key:
            slot = active.index(current_key)
            pct  = int(A_START + slot * slot_w + slot_w / 3)  # 1/3 into slot
            return (_ANALYST_RUNNING_LABEL.get(current_key, f"{current_key} đang phân tích..."), pct)

    # ── E. Msg Clear fired: transition between analysts ───────────────────
    if _last_msg_is_clear(chunk):
        for key in active:
            if not chunk.get(_REPORT_KEY[key]):
                slot = active.index(key)
                pct  = int(A_START + slot * slot_w)
                return (_ANALYST_RUNNING_LABEL.get(key, f"{key} đang bắt đầu..."), pct)

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
        # Never downgrade from completed
        if self.agent_status[agent] == "completed" and status != "completed":
            return
        self.agent_status[agent] = status
        if status == "in_progress":
            self.current_agent = agent

    def update_report_section(self, section_name: str, content: str) -> None:
        if section_name not in self.report_sections:
            return
        self.report_sections[section_name] = content
        # current_report = most recently updated section
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
) -> None:

    mb = MessageBuffer()
    session_mgr.update(session_id, {
        "message_buffer":   mb,
        "status":           "running",
        "current_step":     "Đang khởi tạo đồ thị phân tích",
        "progress_percent": 10,
        "cancel_requested": False,
    })
    _save_sessions_to_disk()

    try:
        if session_mgr.get_field(session_id, "cancel_requested"):
            raise asyncio.CancelledError("Analysis cancelled before start")

        graph = TradingAgentsGraph(debug=True, config=config, selected_analysts=list(analysts))

        # ── Initial agent status ─────────────────────────────────────────
        active_analysts = [k for k in _ANALYST_ORDER if k in analysts]
        for key, name in _ANALYST_DISPLAY.items():
            if key in analysts:
                mb.update_agent_status(name, "pending")
            else:
                mb.agent_status[name] = "not_selected"
        # Mark first analyst in_progress right away
        if active_analysts:
            mb.update_agent_status(_ANALYST_DISPLAY[active_analysts[0]], "in_progress")

        max_debate_rounds = _to_positive_int(config.get("max_debate_rounds"), 1)
        max_risk_rounds   = _to_positive_int(config.get("max_risk_discuss_rounds"), 1)
        last_persist = time.monotonic()

        # ── Deduplication: track which sections have been ingested ───────
        _seen: set = set()

        # ── Previous-state trackers for detecting transitions ────────────
        _prev_invest_cr: str = ""   # investment_debate_state.current_response
        _prev_risk_ls:   str = ""   # risk_debate_state.latest_speaker

        def _on_graph_progress(chunk: Dict[str, Any]) -> None:
            nonlocal last_persist, _prev_invest_cr, _prev_risk_ls

            if session_mgr.get_field(session_id, "cancel_requested"):
                raise asyncio.CancelledError("Analysis cancelled by user")

            # ── 1. Progress bar ──────────────────────────────────────
            info = _derive_realtime_step(chunk, max_debate_rounds, max_risk_rounds, analysts)
            if info is not None:
                step_text, step_pct = info
                prev_pct = int(session_mgr.get_field(session_id, "progress_percent") or 0)
                session_mgr.update(session_id, {
                    "current_step":     step_text,
                    "progress_percent": max(prev_pct, step_pct),
                })

            # ── 2. Tool call → mark analyst in_progress ──────────────
            tool_names = _last_msg_tool_calls(chunk)
            if tool_names:
                analyst_key = _TOOL_ANALYST.get(tool_names[0])
                if analyst_key and analyst_key in active_analysts:
                    disp = _ANALYST_DISPLAY[analyst_key]
                    if mb.agent_status.get(disp) != "completed":
                        mb.update_agent_status(disp, "in_progress")

            # ── 3. Analyst report completions (empty → non-empty) ────
            for key in active_analysts:
                rkey = _REPORT_KEY[key]
                val  = chunk.get(rkey)
                if val and rkey not in _seen:
                    _seen.add(rkey)
                    disp = _ANALYST_DISPLAY[key]
                    mb.update_agent_status(disp, "completed")
                    mb.update_report_section(rkey, val)
                    # Advance to next analyst
                    idx = active_analysts.index(key)
                    if idx + 1 < len(active_analysts):
                        next_disp = _ANALYST_DISPLAY[active_analysts[idx + 1]]
                        mb.update_agent_status(next_disp, "in_progress")
                    else:
                        mb.update_agent_status("AlphaGPT Analyst", "in_progress")

            # ── 4. Msg Clear: next analyst starting ──────────────────
            if _last_msg_is_clear(chunk):
                for key in active_analysts:
                    if not chunk.get(_REPORT_KEY[key]):
                        disp = _ANALYST_DISPLAY[key]
                        if mb.agent_status.get(disp) != "completed":
                            mb.update_agent_status(disp, "in_progress")
                        break

            # ── 5. Quant report ──────────────────────────────────────
            if chunk.get("quant_report") and "quant_report" not in _seen:
                _seen.add("quant_report")
                mb.update_agent_status("AlphaGPT Analyst", "completed")
                mb.update_report_section("quant_report", chunk["quant_report"])
                mb.update_agent_status("Bull Researcher", "in_progress")

            # ── 6. Investment debate state ────────────────────────────
            invest_state = chunk.get("investment_debate_state")
            if isinstance(invest_state, dict):
                cr = str(invest_state.get("current_response", ""))
                jd = str(invest_state.get("judge_decision", ""))
                if cr != _prev_invest_cr and cr:
                    _prev_invest_cr = cr
                    if cr.startswith("Bull Analyst"):
                        mb.update_agent_status("Bull Researcher", "in_progress")
                    elif cr.startswith("Bear Analyst"):
                        # Bull just finished this turn
                        mb.update_agent_status("Bear Researcher", "in_progress")
                if jd and "invest_judge" not in _seen:
                    _seen.add("invest_judge")
                    mb.update_agent_status("Bull Researcher",    "completed")
                    mb.update_agent_status("Bear Researcher",    "completed")
                    mb.update_agent_status("Research Manager",   "in_progress")

            # investment_plan ready
            if chunk.get("investment_plan") and "investment_plan" not in _seen:
                _seen.add("investment_plan")
                mb.update_agent_status("Bull Researcher",    "completed")
                mb.update_agent_status("Bear Researcher",    "completed")
                mb.update_agent_status("Research Manager",   "completed")
                mb.update_report_section("investment_plan", chunk["investment_plan"])
                mb.update_agent_status("Trader", "in_progress")

            # ── 7. Trader plan ───────────────────────────────────────
            if chunk.get("trader_investment_plan") and "trader_investment_plan" not in _seen:
                _seen.add("trader_investment_plan")
                mb.update_agent_status("Trader", "completed")
                mb.update_report_section("trader_investment_plan", chunk["trader_investment_plan"])
                mb.update_agent_status("Risky Analyst", "in_progress")

            # ── 8. Risk debate state ─────────────────────────────────
            risk_state = chunk.get("risk_debate_state")
            if isinstance(risk_state, dict):
                ls = str(risk_state.get("latest_speaker", ""))
                jd = str(risk_state.get("judge_decision", ""))
                if ls != _prev_risk_ls and ls:
                    _prev_risk_ls = ls
                    _transitions = {
                        # current_speaker: (agent_to_mark_in_progress, agent_to_mark_completed)
                        "Risky":   ("Risky Analyst",    None),
                        "Safe":    ("Safe Analyst",     "Risky Analyst"),
                        "Neutral": ("Neutral Analyst",  "Safe Analyst"),
                    }
                    if ls in _transitions:
                        cur, prev = _transitions[ls]
                        if prev:
                            mb.update_agent_status(prev, "completed")
                        mb.update_agent_status(cur, "in_progress")
                if jd and "risk_judge" not in _seen:
                    _seen.add("risk_judge")
                    mb.update_agent_status("Risky Analyst",   "completed")
                    mb.update_agent_status("Safe Analyst",    "completed")
                    mb.update_agent_status("Neutral Analyst", "completed")
                    mb.update_agent_status("Portfolio Manager", "in_progress")

            # ── 9. Final decision ─────────────────────────────────────
            if chunk.get("final_trade_decision") and "final_trade_decision" not in _seen:
                _seen.add("final_trade_decision")
                mb.update_agent_status("Portfolio Manager", "completed")
                mb.update_report_section("final_trade_decision", str(chunk["final_trade_decision"]))

            # ── Periodic disk persist ────────────────────────────────
            now = time.monotonic()
            if now - last_persist >= 1.0:
                _save_sessions_to_disk()
                last_persist = now

        # ── Run graph in thread ──────────────────────────────────────────
        final_state, decision = await asyncio.to_thread(
            lambda: graph.propagate(
                ticker,
                analysis_date,
                alphagpt_signal=alpha_signal,
                progress_callback=_on_graph_progress,
            )
        )

        # ── Final reconciliation: fill anything missed during streaming ──
        _analyst_agent_map = {
            "market_report":       "Market Analyst",
            "sentiment_report":    "Social Analyst",
            "news_report":         "News Analyst",
            "fundamentals_report": "Fundamentals Analyst",
            "quant_report":        "AlphaGPT Analyst",
        }
        for sec_key, agent_name in _analyst_agent_map.items():
            if final_state.get(sec_key):
                if sec_key not in _seen:
                    mb.update_report_section(sec_key, final_state[sec_key])
                mb.update_agent_status(agent_name, "completed")

        if final_state.get("investment_plan"):
            for a in ["Bull Researcher", "Bear Researcher", "Research Manager"]:
                mb.update_agent_status(a, "completed")
            if "investment_plan" not in _seen:
                mb.update_report_section("investment_plan", final_state["investment_plan"])

        if final_state.get("trader_investment_plan"):
            mb.update_agent_status("Trader", "completed")
            if "trader_investment_plan" not in _seen:
                mb.update_report_section("trader_investment_plan", final_state["trader_investment_plan"])

        if final_state.get("final_trade_decision"):
            for a in ["Risky Analyst", "Safe Analyst", "Neutral Analyst", "Portfolio Manager"]:
                mb.update_agent_status(a, "completed")
            if "final_trade_decision" not in _seen:
                mb.update_report_section("final_trade_decision", str(final_state["final_trade_decision"]))

        # Any remaining pending/in_progress → completed
        for agent in list(mb.agent_status.keys()):
            if mb.agent_status[agent] not in {"not_selected", "completed"}:
                mb.update_agent_status(agent, "completed")

        ta_decision_label = _extract_decision_label(decision)
        fused_decision, fusion_note = _fuse_decision_with_alphagpt(ta_decision_label, alpha_signal)

        session_mgr.update(session_id, {
            "status":               "completed",
            "current_step":         "Hoàn thành phân tích",
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
            "status":        "cancelled",
            "current_step":  "Đã hủy phân tích theo yêu cầu",
            "error":         str(e),
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
            "status":        "error",
            "current_step":  "Phân tích thất bại",
            "error":         str(e),
            "error_details": error_details,
        })
        if not session_mgr.get_field(session_id, "progress_percent"):
            session_mgr.set_field(session_id, "progress_percent", 0)
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
        config["deep_think_llm"]    = request.deep_think_llm
        config["quick_think_llm"]   = request.quick_think_llm
        config["max_debate_rounds"] = request.research_depth or request.max_debate_rounds
        config["data_vendors"]      = request.data_vendors

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
            session_id=session_id,
            status="started",
            message=f"Analysis started for {request.ticker}",
        )

    except Exception as e:
        import traceback
        logger.error("Lỗi start_analysis: %s", traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "details": traceback.format_exc()},
        )


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

    # Session loaded from disk
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
        "current_step":     "Đang hủy phân tích...",
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


@app.get("/api/config/section-titles")
async def get_section_titles():
    return {"section_titles": SECTION_TITLES}


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def on_startup():
    _load_sessions_from_disk()
    _init_vnstock_auth()
    try:
        init_alpha_manager()
        logger.info("Alpha manager đã được khởi tạo.")
    except Exception as exc:
        logger.error("Không thể khởi tạo alpha manager: %s", exc, exc_info=True)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
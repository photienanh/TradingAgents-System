"""
app/main.py — TradingAgents FastAPI entrypoint.

Refactored: business logic extracted to services/, routes split to routes/.
"""

import datetime
import asyncio
import logging
from pathlib import Path

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

from tradingagents.default_config import DEFAULT_CONFIG

from app.routes.market import router as market_router
from app.routes.alpha import router as alpha_router
from app.services.session_manager import SessionManager
from app.services.session_serialization import (
    build_persistable_session,
    rebuild_reports_from_final_state,
    rebuild_agent_status_from_final_state,
    SECTION_TITLES,
    DEFAULT_AGENT_NAMES,
)
from app.storage.session_store import SQLiteSessionStore
from app.services.analysis_runner import run_trading_analysis
from alpha.manager import (
    init_alpha_manager,
    trigger_if_needed,
    trigger_if_needed_blocking,
    get_signal_for_ticker,
    get_status,
)

load_dotenv()
logger = logging.getLogger(__name__)

# ── App setup ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="TradingAgents API",
    description="Multi-Agents LLM Financial Trading Framework",
    version="1.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])
app.include_router(market_router)
app.include_router(alpha_router)
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# ── Storage setup ──────────────────────────────────────────────────────────

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


def _init_vnstock_auth() -> None:
    import os
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


# ── Pydantic models ────────────────────────────────────────────────────────

class AnalysisRequest(BaseModel):
    ticker: str
    analysis_date: Optional[str] = None
    trading_horizon: Optional[str] = "short"
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
    alpha_signal: Optional[Dict[str, Any]] = None
    messages: List[Dict[str, Any]] = []
    tool_calls: List[Dict[str, Any]] = []
    error: Optional[str] = None
    error_details: Optional[str] = None


# ── Routes ─────────────────────────────────────────────────────────────────

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
        alpha_signal  = get_signal_for_ticker(request.ticker)
        analysis_date = request.analysis_date or datetime.datetime.now().strftime("%Y-%m-%d")

        config = DEFAULT_CONFIG.copy()
        config["deep_think_llm"]    = request.deep_think_llm
        config["quick_think_llm"]   = request.quick_think_llm
        config["max_debate_rounds"] = request.research_depth or request.max_debate_rounds
        config["data_vendors"]      = request.data_vendors

        session_mgr.set(session_id, {
            "ticker":           request.ticker,
            "analysis_date":    analysis_date,
            "trading_horizon":  request.trading_horizon or "short", 
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
            session_id, request.ticker, analysis_date, config,
            selected_analysts, alpha_signal,
            session_mgr, _save_sessions_to_disk, request.trading_horizon or "short",
        )
        return AnalysisResponse(
            session_id=session_id,
            status="started",
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
            alpha_signal=session.get("alpha_signal"),
            messages=list(mb.messages),
            tool_calls=list(mb.tool_calls),
            error=session.get("error"),
            error_details=session.get("error_details"),
        )

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
    session_mgr.update(session_id, {"cancel_requested": True, "current_step": "Đang hủy phân tích..."})
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


# ── Startup ────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def on_startup():
    _load_sessions_from_disk()
    _init_vnstock_auth()
    try:
        init_alpha_manager(store=session_store)
        logger.info("Alpha manager đã được khởi tạo.")
        logger.info("Alpha daily startup run: waiting for completion...")
        daily_res = await asyncio.to_thread(trigger_if_needed_blocking)
        logger.info("Alpha daily startup result: %s", daily_res)
    except Exception as exc:
        logger.error("Không thể khởi tạo alpha manager: %s", exc, exc_info=True)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
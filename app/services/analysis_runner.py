"""
app/services/analysis_runner.py
Background analysis task — extracted from app/main.py.
"""
from __future__ import annotations

import asyncio
import time
import logging
from typing import Any, Callable, Dict, List, Optional

from app.services.progress_tracker import (
    MessageBuffer,
    ANALYST_ORDER, ANALYST_DISPLAY, REPORT_KEY,
    derive_realtime_step, last_msg_tool_calls, last_msg_is_clear,
    to_positive_int, TOOL_ANALYST,
)

logger = logging.getLogger(__name__)

# ── Analyst display helpers (re-exported for graph setup) ─────────────────

_ANALYST_AGENT_MAP = {
    "market_report":       "Market Analyst",
    "sentiment_report":    "Social Analyst",
    "news_report":         "News Analyst",
    "fundamentals_report": "Fundamentals Analyst",
    "quant_report":        "AlphaGPT Analyst",
}


async def run_trading_analysis(
    session_id: str,
    ticker: str,
    analysis_date: str,
    config: dict,
    analysts: List[str],
    alpha_signal: Optional[Dict[str, Any]],
    session_mgr,
    save_fn: Callable,
    trading_horizon: str = "short",
) -> None:
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    mb = MessageBuffer()
    session_mgr.update(session_id, {
        "message_buffer":   mb,
        "status":           "running",
        "cancel_requested": False,
    })
    save_fn()

    try:
        if session_mgr.get_field(session_id, "cancel_requested"):
            raise asyncio.CancelledError("Analysis cancelled before start")

        graph = TradingAgentsGraph(debug=True, config=config, selected_analysts=list(analysts))

        active_analysts = [k for k in ANALYST_ORDER if k in analysts]
        for key, name in ANALYST_DISPLAY.items():
            if key in analysts:
                mb.update_agent_status(name, "pending")
            else:
                mb.agent_status[name] = "not_selected"
        if active_analysts:
            mb.update_agent_status(ANALYST_DISPLAY[active_analysts[0]], "in_progress")

        max_debate_rounds = to_positive_int(config.get("max_debate_rounds"), 1)
        max_risk_rounds   = to_positive_int(config.get("max_risk_discuss_rounds"), 1)
        last_persist = time.monotonic()
        _seen: set = set()
        _prev_invest_cr: str = ""
        _prev_risk_ls:   str = ""

        def _on_graph_progress(chunk: Dict[str, Any]) -> None:
            nonlocal last_persist, _prev_invest_cr, _prev_risk_ls

            if session_mgr.get_field(session_id, "cancel_requested"):
                raise asyncio.CancelledError("Analysis cancelled by user")

            # Tool call → mark analyst in_progress
            tool_names = last_msg_tool_calls(chunk)
            if tool_names:
                analyst_key = TOOL_ANALYST.get(tool_names[0])
                if analyst_key and analyst_key in active_analysts:
                    disp = ANALYST_DISPLAY[analyst_key]
                    if mb.agent_status.get(disp) != "completed":
                        mb.update_agent_status(disp, "in_progress")

            # Analyst report completions
            for key in active_analysts:
                rkey = REPORT_KEY[key]
                val  = chunk.get(rkey)
                if val and rkey not in _seen:
                    _seen.add(rkey)
                    disp = ANALYST_DISPLAY[key]
                    mb.update_agent_status(disp, "completed")
                    mb.update_report_section(rkey, val)
                    idx = active_analysts.index(key)
                    if idx + 1 < len(active_analysts):
                        next_disp = ANALYST_DISPLAY[active_analysts[idx + 1]]
                        mb.update_agent_status(next_disp, "in_progress")
                    else:
                        if "alpha" in analysts:
                            mb.update_agent_status("AlphaGPT Analyst", "in_progress")
                        else:
                            mb.update_agent_status("Bull Researcher", "in_progress")

            if last_msg_is_clear(chunk):
                for key in active_analysts:
                    if not chunk.get(REPORT_KEY[key]):
                        disp = ANALYST_DISPLAY[key]
                        if mb.agent_status.get(disp) != "completed":
                            mb.update_agent_status(disp, "in_progress")
                        break

            if chunk.get("quant_report") and "quant_report" not in _seen:
                _seen.add("quant_report")
                mb.update_agent_status("AlphaGPT Analyst", "completed")
                mb.update_report_section("quant_report", chunk["quant_report"])
                mb.update_agent_status("Bull Researcher", "in_progress")

            invest_state = chunk.get("investment_debate_state")
            if isinstance(invest_state, dict):
                cr = str(invest_state.get("current_response", ""))
                jd = str(invest_state.get("judge_decision", ""))
                if cr != _prev_invest_cr and cr:
                    _prev_invest_cr = cr
                    if cr.startswith("Bull Analyst"):
                        mb.update_agent_status("Bull Researcher", "in_progress")
                    elif cr.startswith("Bear Analyst"):
                        mb.update_agent_status("Bear Researcher", "in_progress")
                if jd and "invest_judge" not in _seen:
                    _seen.add("invest_judge")
                    mb.update_agent_status("Bull Researcher",  "completed")
                    mb.update_agent_status("Bear Researcher",  "completed")
                    mb.update_agent_status("Research Manager", "in_progress")

            if chunk.get("investment_plan") and "investment_plan" not in _seen:
                _seen.add("investment_plan")
                mb.update_agent_status("Bull Researcher",  "completed")
                mb.update_agent_status("Bear Researcher",  "completed")
                mb.update_agent_status("Research Manager", "completed")
                mb.update_report_section("investment_plan", chunk["investment_plan"])
                mb.update_agent_status("Trader", "in_progress")

            if chunk.get("trader_investment_plan") and "trader_investment_plan" not in _seen:
                _seen.add("trader_investment_plan")
                mb.update_agent_status("Trader", "completed")
                mb.update_report_section("trader_investment_plan", chunk["trader_investment_plan"])
                mb.update_agent_status("Risky Analyst", "in_progress")

            risk_state = chunk.get("risk_debate_state")
            if isinstance(risk_state, dict):
                ls = str(risk_state.get("latest_speaker", ""))
                jd = str(risk_state.get("judge_decision", ""))
                if ls != _prev_risk_ls and ls:
                    _prev_risk_ls = ls
                    _transitions = {
                        "Risky":   ("Risky Analyst",   None),
                        "Safe":    ("Safe Analyst",    "Risky Analyst"),
                        "Neutral": ("Neutral Analyst", "Safe Analyst"),
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

            if chunk.get("final_trade_decision") and "final_trade_decision" not in _seen:
                _seen.add("final_trade_decision")
                mb.update_agent_status("Portfolio Manager", "completed")
                mb.update_report_section("final_trade_decision", str(chunk["final_trade_decision"]))

            now = time.monotonic()
            if now - last_persist >= 1.0:
                save_fn()
                last_persist = now

        final_state, decision = await asyncio.to_thread(
            lambda: graph.propagate(
                ticker, analysis_date,
                alphagpt_signal=alpha_signal,
                progress_callback=_on_graph_progress,
                trading_horizon=trading_horizon,
            )
        )

        # Final reconciliation
        for sec_key, agent_name in _ANALYST_AGENT_MAP.items():
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

        for agent in list(mb.agent_status.keys()):
            if mb.agent_status[agent] not in {"not_selected", "completed"}:
                mb.update_agent_status(agent, "completed")

        session_mgr.update(session_id, {
            "status":               "completed",
            "decision":             decision,
            "alpha_signal":         alpha_signal,
            "final_state":          final_state,
        })
        save_fn()

    except asyncio.CancelledError as e:
        session_mgr.update(session_id, {
            "status":        "cancelled",
            "error":         str(e),
            "error_details": None,
        })
        save_fn()

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logger.error("Lỗi phân tích %s: %s", session_id, error_details)
        session_mgr.update(session_id, {
            "status":        "error",
            "error":         str(e),
            "error_details": error_details,
        })
        mb.add_message("Error", f"Analysis failed: {str(e)}")
        save_fn()
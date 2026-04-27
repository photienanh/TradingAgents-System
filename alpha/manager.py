"""
alpha/manager.py

Runtime manager used by FastAPI app and TradingAgents integration.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Optional

from alpha.daily_runner import (
    MARKET_DATA_DIR,
    SIGNALS_PATH,
    run_daily_update,
    _safe_float,
)

log = logging.getLogger(__name__)

_STATE_LOCK = threading.Lock()
_STATUS: Dict[str, Any] = {
    "running": False,
    "last_run_day": None,
    "last_run_at": None,
    "last_error": None,
}
_SIGNALS_CACHE: Dict[str, Dict[str, Any]] = {}
DAILY_START_HOUR = int(os.getenv("ALPHAGPT_DAILY_START_HOUR", "9"))


def _is_after_daily_start(now: datetime) -> bool:
    return now.hour >= DAILY_START_HOUR


def _load_latest_cache() -> None:
    global _SIGNALS_CACHE
    if not SIGNALS_PATH.exists():
        return
    try:
        import pandas as pd
        df = pd.read_csv(SIGNALS_PATH)
        for _, row in df.iterrows():
            t = str(row["ticker"]).upper()
            _SIGNALS_CACHE[t] = {
                "enabled":      bool(row.get("enabled", False)),
                "ticker":       t,
                "side":         str(row.get("side", "neutral")),
                "signal_today": _safe_float(row.get("signal_today")),
                "ic_oos":       _safe_float(row.get("ic_oos")),
                "sharpe_oos":   _safe_float(row.get("sharpe_oos")),
                "return_oos":   _safe_float(row.get("return_oos")),
                "as_of":        str(row.get("as_of", "")),
            }
    except Exception as exc:
        log.warning("[AlphaManager] Failed loading cache %s: %s", SIGNALS_PATH, exc)


def _run_daily_task() -> None:
    with _STATE_LOCK:
        if _STATUS["running"]:
            return
        _STATUS["running"] = True
        _STATUS["last_error"] = None

    try:
        result = run_daily_update()
        _load_latest_cache()
        with _STATE_LOCK:
            _STATUS["last_run_day"] = date.today().isoformat()
            _STATUS["last_run_at"] = datetime.now().isoformat()
            _STATUS["last_result"] = result
    except Exception as exc:
        log.exception("[AlphaManager] Daily update failed")
        with _STATE_LOCK:
            _STATUS["last_error"] = str(exc)
    finally:
        with _STATE_LOCK:
            _STATUS["running"] = False

def _check_should_skip() -> Optional[Dict[str, Any]]:
    """Trả về dict nếu nên skip, None nếu được phép chạy."""
    with _STATE_LOCK:
        running  = _STATUS["running"]
        last_day = _STATUS.get("last_run_day")

    if running:
        return {"accepted": False, "message": "Alpha daily update already running"}

    now = datetime.now()
    if not _is_after_daily_start(now):
        return {
            "accepted": False,
            "message": f"Alpha daily update is scheduled after {DAILY_START_HOUR:02d}:00",
            "scheduled_after": f"{DAILY_START_HOUR:02d}:00",
        }

    if last_day == date.today().isoformat():
        return {"accepted": False, "message": "Alpha daily update already completed today"}

    return None

def trigger_if_needed(force: bool = False) -> Dict[str, Any]:
    if not force:
        skip = _check_should_skip()
        if skip:
            return skip
    thread = threading.Thread(target=_run_daily_task, daemon=True, name="alpha-daily-runner")
    thread.start()
    return {"accepted": True, "message": "Alpha daily update started"}


def trigger_if_needed_blocking(force: bool = False) -> Dict[str, Any]:
    if not force:
        skip = _check_should_skip()
        if skip:
            return skip
    _run_daily_task()
    with _STATE_LOCK:
        err    = _STATUS.get("last_error")
        result = _STATUS.get("last_result")
    if err:
        return {"accepted": False, "message": "Alpha daily update failed", "error": err}
    return {"accepted": True, "message": "Alpha daily update finished", "result": result}


def get_status() -> Dict[str, Any]:
    with _STATE_LOCK:
        status = dict(_STATUS)
    status["n_cached_signals"] = len(_SIGNALS_CACHE)
    status["market_data_tickers"] = len(list(MARKET_DATA_DIR.glob("*.csv"))) if MARKET_DATA_DIR.exists() else 0
    return status


def _ensure_cache_loaded() -> None:
    if not _SIGNALS_CACHE:
        _load_latest_cache()


def get_signal_for_ticker(ticker: str) -> Dict[str, Any]:
    t = ticker.upper().strip()
    _ensure_cache_loaded()

    if t in _SIGNALS_CACHE:
        payload = dict(_SIGNALS_CACHE[t])
        payload.setdefault("ticker", t)
        return payload
    
    return {
        "enabled": False,
        "ticker": t,
        "side": "neutral",
        "signal_today": None,
        "error": "Ticker signal not found in daily cache",
    }


def get_all_signals(limit: Optional[int] = None) -> Dict[str, Dict[str, Any]]:
    _ensure_cache_loaded()
    if limit is None or limit <= 0:
        return dict(_SIGNALS_CACHE)

    items = list(_SIGNALS_CACHE.items())[:limit]
    return {k: v for k, v in items}


def init_alpha_manager() -> Dict[str, Any]:
    """Compatibility init hook used by FastAPI startup."""
    _load_latest_cache()
    return get_status()

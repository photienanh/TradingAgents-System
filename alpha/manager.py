"""
alpha/manager.py

Runtime manager used by FastAPI app and TradingAgents integration.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Optional

from alpha.daily_runner import (
    MARKET_DATA_DIR,
    SIGNALS_DIR,
    compute_ticker_signal,
    get_top_alphas_by_ic_oos,
    run_daily_update,
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


def _latest_signal_file() -> Optional[Path]:
    if not SIGNALS_DIR.exists():
        return None
    files = sorted(SIGNALS_DIR.glob("alpha_signals_*.json"), reverse=True)
    return files[0] if files else None


def _load_latest_cache() -> None:
    global _SIGNALS_CACHE
    path = _latest_signal_file()
    if not path:
        return
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            _SIGNALS_CACHE = {str(k).upper(): v for k, v in data.items()}
    except Exception as exc:
        log.warning("[AlphaManager] Failed loading cache %s: %s", path, exc)


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


def trigger_daily_update_async(force: bool = False) -> Dict[str, Any]:
    """Start daily runner in background; skip if already up-to-date unless force=True."""
    with _STATE_LOCK:
        running = _STATUS["running"]
        last_day = _STATUS.get("last_run_day")

    if running:
        return {"accepted": False, "message": "Alpha daily update already running"}

    today = date.today().isoformat()
    if not force and last_day == today:
        return {"accepted": False, "message": "Alpha daily update already completed today"}

    thread = threading.Thread(target=_run_daily_task, daemon=True, name="alpha-daily-runner")
    thread.start()
    return {"accepted": True, "message": "Alpha daily update started"}


def trigger_daily_update_if_needed() -> Dict[str, Any]:
    return trigger_daily_update_async(force=False)


def get_alpha_status() -> Dict[str, Any]:
    with _STATE_LOCK:
        status = dict(_STATUS)
    status["n_cached_signals"] = len(_SIGNALS_CACHE)
    status["market_data_tickers"] = len(list(MARKET_DATA_DIR.glob("*.csv"))) if MARKET_DATA_DIR.exists() else 0
    return status


def _ensure_cache_loaded() -> None:
    if not _SIGNALS_CACHE:
        _load_latest_cache()


def get_signal_for_ticker(ticker: str, recompute_if_missing: bool = True) -> Dict[str, Any]:
    t = ticker.upper().strip()
    _ensure_cache_loaded()

    if t in _SIGNALS_CACHE:
        payload = dict(_SIGNALS_CACHE[t])
        payload.setdefault("ticker", t)
        return payload

    if recompute_if_missing:
        top_alphas = get_top_alphas_by_ic_oos(limit=5)
        payload = compute_ticker_signal(t, top_alphas=top_alphas)
        _SIGNALS_CACHE[t] = payload
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
    return get_alpha_status()


def trigger_if_needed() -> Dict[str, Any]:
    """Compatibility wrapper used by app.main."""
    return trigger_daily_update_if_needed()


def get_status() -> Dict[str, Any]:
    """Compatibility wrapper used by app.main health endpoint."""
    return get_alpha_status()

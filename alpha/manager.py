"""
alpha/manager.py

Singleton quản lý vòng đời alpha pipeline trong FastAPI app.
main.py chỉ cần gọi 3 hàm:
  init_alpha_manager()       — gọi trong startup event
  trigger_if_needed()        — gọi trước mỗi analysis request
  get_signal_for_ticker(t)   — lấy signal cho TradingAgents

Thiết kế:
- Không duplicate logic của daily_runner.py
- Chạy daily_runner.main() trong background thread khi cần
- PipelineState đảm bảo chỉ chạy 1 lần/ngày kể cả qua server restart
"""
from __future__ import annotations

import logging
import threading
from datetime import datetime
from typing import Any, Dict, Optional

from alpha.core.paths import SIGNALS_DIR, BASE_DIR
from alpha.core.signal_store import SignalStore
from alpha.pipelines.pipeline_state import PipelineState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton state
# ---------------------------------------------------------------------------
_signal_store: Optional[SignalStore] = None
_pipeline_state: Optional[PipelineState] = None
_running = False
_lock = threading.Lock()

# Giờ mở cửa thị trường — chỉ trigger pipeline sau thời điểm này
_MARKET_OPEN_HOUR = 9


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def init_alpha_manager() -> None:
    """Gọi một lần trong FastAPI startup event."""
    global _signal_store, _pipeline_state

    import os
    state_dir = os.path.join(str(BASE_DIR), "alpha", "pipelines")
    state_db = os.path.join(str(BASE_DIR), "app", "data", "sessions.db")

    _signal_store   = SignalStore(str(SIGNALS_DIR))
    _pipeline_state = PipelineState(state_dir=state_dir, db_path=state_db)

    triggered = trigger_if_needed()
    if triggered:
        logger.info("[AlphaManager] Daily pipeline triggered khi startup.")
    else:
        logger.info(
            "[AlphaManager] Pipeline không trigger. Lý do: %s",
            _reason_not_triggered(),
        )


def trigger_if_needed() -> bool:
    """
    Trigger daily pipeline nếu:
    - Đã qua 9h sáng
    - Chưa chạy hôm nay (check cả qua restart)
    - Không đang chạy

    Thread-safe, idempotent.
    Trả về True nếu pipeline được kích hoạt.
    """
    global _running

    if not _should_run():
        return False

    with _lock:
        if not _should_run():
            return False
        _running = True

    thread = threading.Thread(
        target=_run_pipeline_safe,
        name="alpha-daily-pipeline",
        daemon=True,
    )
    thread.start()
    logger.info("[AlphaManager] Background pipeline thread started.")
    return True


def get_signal_for_ticker(ticker: str) -> Dict[str, Any]:
    """Trả về signal dict cho TradingAgents."""
    if _signal_store is None:
        return {
            "enabled": False,
            "ticker":  ticker.upper(),
            "side":    "neutral",
            "score":   None,
            "rank":    None,
            "error":   "Alpha manager chưa được khởi tạo",
        }
    return _signal_store.get_signal_for_ticker(ticker)


def get_status() -> Dict[str, Any]:
    """Trả về status cho health check endpoint."""
    return {
        "initialized":      _signal_store is not None,
        "is_running":       _running,
        "should_run":       _should_run(),
        "signal_fresh":     _signal_store.is_fresh(1) if _signal_store else False,
        "last_run_date":    str(_pipeline_state.last_run_date) if _pipeline_state else None,
        "already_ran_today": _pipeline_state.already_ran_today() if _pipeline_state else False,
    }


# ---------------------------------------------------------------------------
# Private
# ---------------------------------------------------------------------------

def _should_run() -> bool:
    global _running
    if _running:
        return False
    if datetime.now().hour < _MARKET_OPEN_HOUR:
        return False
    if _pipeline_state is None:
        return False
    if _pipeline_state.already_ran_today():
        return False
    return True


def _run_pipeline_safe() -> None:
    """Wrapper chạy daily_runner trong background thread."""
    global _running
    try:
        _run_pipeline()
    except Exception as exc:
        logger.error("[AlphaManager] Pipeline thất bại: %s", exc, exc_info=True)
    finally:
        _running = False


def _run_pipeline() -> None:
    """
    Gọi trực tiếp các hàm của daily_runner thay vì subprocess.
    Tương đương chạy: python daily_runner.py --all
    """
    from alpha.core.daily_runner import (
        refresh_latest_market_data,
        refresh_latest_sentiment_data,
        refresh_alpha_values_from_existing_formulas,
        run_daily,
    )
    from alpha.core.universe import VN30_SYMBOLS
    from datetime import date

    tickers = VN30_SYMBOLS
    logger.info("[AlphaManager] Bắt đầu daily pipeline cho %d tickers", len(tickers))

    # 1. Market data
    market_stats = refresh_latest_market_data(tickers)
    target_day_str = market_stats.get("target_day")
    target_day = None
    if target_day_str:
        import pandas as pd
        target_day = pd.Timestamp(target_day_str).normalize()
    logger.info(
        "[AlphaManager] Market refresh: target=%s updated=%d skipped=%d failed=%d",
        target_day_str,
        len(market_stats.get("updated", [])),
        len(market_stats.get("skipped", [])),
        len(market_stats.get("failed", [])),
    )

    # 2. Sentiment
    sent_stats = refresh_latest_sentiment_data(tickers, target_day)
    logger.info(
        "[AlphaManager] Sentiment refresh: crawled=%d new_news=%d",
        sent_stats.get("symbols_crawled", 0),
        sent_stats.get("news_rows_added", 0),
    )

    # 3. Alpha values
    alpha_stats = refresh_alpha_values_from_existing_formulas(tickers)
    logger.info(
        "[AlphaManager] Alpha refresh: updated=%d failed=%d",
        len(alpha_stats.get("updated", [])),
        len(alpha_stats.get("failed", [])),
    )

    # 4. Compute signals
    result = run_daily(tickers)
    if result.empty:
        logger.warning("[AlphaManager] Không tạo được signal nào!")
    else:
        logger.info(
            "[AlphaManager] Signals: %d tickers (BUY=%d SELL=%d HOLD=%d)",
            len(result),
            (result["action"] == "BUY").sum(),
            (result["action"] == "SELL").sum(),
            (result["action"] == "HOLD").sum(),
        )

    # 5. Lưu state
    _pipeline_state.last_run_date = date.today()
    logger.info("[AlphaManager] Daily pipeline hoàn tất: %s", date.today())


def _reason_not_triggered() -> str:
    if _running:
        return "đang chạy"
    if _pipeline_state and _pipeline_state.already_ran_today():
        return "đã chạy hôm nay"
    if datetime.now().hour < _MARKET_OPEN_HOUR:
        return f"chưa đến {_MARKET_OPEN_HOUR}h sáng"
    return "không rõ"
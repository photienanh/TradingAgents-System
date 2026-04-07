"""
alpha/core/signal_store.py

Đọc signal từ data/signals/signals_{YYYYMMDD}.csv mới nhất.
Serve signal cho TradingAgents qua get_signal_for_ticker().

Lưu ý thiết kế:
- KHÔNG build/save signal ở đây — đó là việc của daily_runner.run_daily()
- Ngưỡng BUY/SELL dùng ±1.0 (z-score) khớp với daily_runner
- Thread-safe cho FastAPI
"""
from __future__ import annotations

import logging
import threading
from datetime import date
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# action → side mapping cho TradingAgents
_ACTION_TO_SIDE = {"BUY": "long", "SELL": "short", "HOLD": "neutral"}


class SignalStore:
    """
    Thread-safe reader cho data/signals/signals_{YYYYMMDD}.csv.
    Chỉ đọc — không ghi. daily_runner.run_daily() chịu trách nhiệm ghi.
    """

    def __init__(self, signals_dir: str):
        self.signals_dir = Path(signals_dir)
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_signal_for_ticker(self, ticker: str) -> Dict[str, Any]:
        """
        Trả về signal dict cho TradingAgents.
        Interface: enabled, ticker, side, score, rank, action, strength, avg_ic, avg_sharpe
        """
        ticker_upper = ticker.strip().upper()
        base: Dict[str, Any] = {
            "enabled": False,
            "ticker":  ticker_upper,
            "side":    "neutral",
            "score":   None,
            "rank":    None,
        }

        df = self.load_latest()
        if df is None or df.empty:
            base["error"] = "Chưa có signal file nào trong data/signals/"
            return base

        # Freshness check
        self._check_freshness(base, df)

        match = df[df["ticker"].astype(str).str.upper() == ticker_upper]
        if match.empty:
            base["error"] = f"{ticker_upper} không có trong signal mới nhất"
            return base

        row    = match.iloc[0]
        action = str(row.get("action", "HOLD")).strip().upper()
        side   = _ACTION_TO_SIDE.get(action, "neutral")

        result: Dict[str, Any] = {
            "enabled":    True,
            "ticker":     ticker_upper,
            "side":       side,
            "action":     action,
            "score":      _to_float(row.get("signal")),
            "rank":       _to_int(row.get("rank")),
            "strength":   _to_int(row.get("strength")),
            "avg_ic":     _to_float(row.get("avg_ic")),
            "avg_sharpe": _to_float(row.get("avg_sharpe")),
        }
        if "warning" in base:
            result["warning"] = base["warning"]
        return result

    def load_latest(self) -> Optional[pd.DataFrame]:
        """Đọc file signals mới nhất (theo tên file, không phải mtime)."""
        with self._lock:
            files = sorted(self.signals_dir.glob("signals_*.csv"), reverse=True)
            if not files:
                return None
            try:
                df = pd.read_csv(files[0])
                logger.debug("Đọc signals từ %s (%d dòng)", files[0].name, len(df))
                return df
            except Exception as exc:
                logger.error("Đọc signal file lỗi: %s", exc)
                return None

    def load_for_date(self, trade_date: date) -> Optional[pd.DataFrame]:
        """Đọc file signals cho một ngày cụ thể."""
        path = self.signals_dir / f"signals_{trade_date.strftime('%Y%m%d')}.csv"
        if not path.exists():
            return None
        try:
            return pd.read_csv(path)
        except Exception as exc:
            logger.error("Đọc %s lỗi: %s", path.name, exc)
            return None

    def is_fresh(self, max_age_days: int = 1) -> bool:
        """True nếu file signal mới nhất không cũ hơn max_age_days."""
        with self._lock:
            files = sorted(self.signals_dir.glob("signals_*.csv"), reverse=True)
            if not files:
                return False
            try:
                date_part = files[0].stem.replace("signals_", "")
                file_date = pd.to_datetime(date_part, format="%Y%m%d").date()
                return (date.today() - file_date).days <= max_age_days
            except Exception:
                return False

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    @staticmethod
    def _check_freshness(base: Dict[str, Any], df: pd.DataFrame) -> None:
        """Thêm warning vào base nếu signal cũ hơn 5 ngày."""
        try:
            if "date" in df.columns:
                last = pd.to_datetime(df["date"].iloc[0]).date()
                days_old = (date.today() - last).days
                if days_old > 5:
                    base["warning"] = f"Signal cũ {days_old} ngày (cập nhật: {last})"
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_float(val: Any) -> Optional[float]:
    if val is None:
        return None
    try:
        v = float(val)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _to_int(val: Any) -> Optional[int]:
    f = _to_float(val)
    return int(f) if f is not None else None
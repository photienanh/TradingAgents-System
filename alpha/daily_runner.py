"""
alpha/daily_runner.py

Daily updater for AlphaGPT integration:
- Refresh market data CSV files (best-effort, incremental)
- Tính toán technical indicators và lưu vào cùng file CSV
- Build per-ticker alpha bias from top-5 alphas in data/alpha_library.json (sorted by ic_oos)
- Persist daily signal snapshot to data/signals/
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from alpha import alpha_operators as op
from alpha.data_loader import add_technical_indicators

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MARKET_DATA_DIR = DATA_DIR / "market_data"
ALPHA_LIBRARY_PATH = DATA_DIR / "alpha_library.json"
SIGNALS_DIR = DATA_DIR / "signals"
SIGNALS_PATH = SIGNALS_DIR / "alpha_signals.csv"

SLIGHTLY_LONG_THRESHOLD = 0.30
SLIGHTLY_SHORT_THRESHOLD = -0.30
LONG_THRESHOLD = 0.80
SHORT_THRESHOLD = -0.80

VNSTOCK_FALLBACK_TICKERS = {
    "ASG", "CLC", "CMV", "CVT", "GTA", "HTV", "HU1", "NAV", "NVT",
    "OPC", "PGI", "PNC", "S4A", "SSC", "STG", "TDM", "TMP", "TPC",
    "TVT", "UIC", "VAB", "VAF",
}

# Số mã đầu tiên kiểm tra để phát hiện ngày nghỉ
HOLIDAY_PROBE_COUNT = 5


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        out = float(value)
        return out if np.isfinite(out) else None
    except (TypeError, ValueError):
        return None


def _load_alpha_library(path: Path = ALPHA_LIBRARY_PATH) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
    except Exception as exc:
        log.warning("[DailyRunner] Failed to read alpha library: %s", exc)
    return []


def load_alpha_definitions(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    alphas = _load_alpha_library()
    valid: List[Dict[str, Any]] = []
    for alpha in alphas:
        formula = str(alpha.get("formula", "")).strip()
        ic_oos = _safe_float(alpha.get("ic_oos"))
        if not formula or ic_oos is None:
            continue
        description = str(alpha.get("description", ""))
        if description.startswith("GP"):
            description = description.split(': ', 1)[1]
        valid.append({
            "id":          str(alpha.get("id", "")),
            "description": description,
            "hypothesis":  str(alpha.get("hypothesis", "")),
            "formula":     formula,
            "ic_oos":      ic_oos,
            "sharpe_oos":  _safe_float(alpha.get("sharpe_oos")),
            "return_oos":  _safe_float(alpha.get("return_oos")),
        })
    valid.sort(key=lambda a: a["ic_oos"], reverse=True)
    return valid if limit is None else valid[:limit]


def _load_ticker_frame(ticker: str, market_data_dir: Path = MARKET_DATA_DIR) -> Optional[pd.DataFrame]:
    path = market_data_dir / f"{ticker.upper()}.csv"
    if not path.exists():
        return None

    try:
        raw = pd.read_csv(path)
        raw.columns = [c.lower() for c in raw.columns]
        if "time" not in raw.columns:
            return None

        raw["time"] = pd.to_datetime(raw["time"], errors="coerce").dt.normalize()
        raw = raw.dropna(subset=["time", "open", "high", "low", "close", "volume"])
        raw = raw.sort_values("time").set_index("time")

        # Nếu file đã có indicators thì dùng luôn, không tính lại
        indicator_cols = [
            "sma_5", "sma_20", "sma_50", "sma_200", "ema_10",
            "rsi_14", "macd", "macd_signal", "macd_hist",
            "bb_upper", "bb_middle", "bb_lower",
            "atr_14", "vwma_20", "mfi_14",
        ]
        has_indicators = all(c in raw.columns for c in ["rsi_14", "macd", "bb_upper"])

        if has_indicators:
            # Chỉ giữ các cột cần thiết, không tính lại
            keep = ["open", "high", "low", "close", "volume"]
            if "industry" in raw.columns:
                keep.append("industry")
            # Thêm các cột indicator đã có
            for c in indicator_cols + ["vwap", "adv20", "obv", "returns", "momentum_3", "momentum_10"]:
                if c in raw.columns:
                    keep.append(c)
            df = raw[[c for c in keep if c in raw.columns]]
        else:
            keep = ["open", "high", "low", "close", "volume"]
            if "industry" in raw.columns:
                keep.append("industry")
            base = raw[[c for c in keep if c in raw.columns]]
            df = add_technical_indicators(base)

        return df
    except Exception as exc:
        log.warning("[DailyRunner] Failed to load %s: %s", ticker, exc)
        return None


def _normalize_yf_ticker(symbol: str) -> str:
    ticker = symbol.strip().upper()
    return ticker if ticker.endswith(".VN") else f"{ticker}.VN"


def _fetch_ohlcv_yfinance(symbol: str, start: str, end: str) -> pd.DataFrame:
    ticker = yf.Ticker(_normalize_yf_ticker(symbol))
    end_plus_one = (datetime.strptime(end, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    history = ticker.history(start=start, end=end_plus_one)
    if history is None or history.empty:
        return pd.DataFrame()

    history = history.copy()
    history.reset_index(inplace=True)
    history.rename(
        columns={
            "Date": "time",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        },
        inplace=True,
    )
    keep_cols = ["time", "open", "high", "low", "close", "volume"]
    history = history[keep_cols]
    if getattr(history["time"].dt, "tz", None) is not None:
        history["time"] = history["time"].dt.tz_localize(None)
    history["time"] = pd.to_datetime(history["time"], errors="coerce").dt.normalize()
    history = history.dropna(subset=["time", "open", "high", "low", "close", "volume"])
    return history.sort_values("time").reset_index(drop=True)


def _fetch_ohlcv_vnstock(symbol: str, start: str, end: str) -> pd.DataFrame:
    try:
        from vnstock import Quote
    except Exception:
        return pd.DataFrame()

    try:
        history = Quote(symbol=symbol, source="KBS").history(start=start, end=end)
    except Exception:
        return pd.DataFrame()

    if history is None or history.empty:
        return pd.DataFrame()

    history = history.copy()
    history.reset_index(inplace=True)
    rename_map = {
        "Date": "time", "Open": "open", "High": "high",
        "Low": "low", "Close": "close", "Volume": "volume",
    }
    history.rename(columns=rename_map, inplace=True)
    if "ticker" not in history.columns:
        history["ticker"] = symbol.upper()
    if "industry" not in history.columns:
        history["industry"] = pd.NA
    history = history[[c for c in ["time", "ticker", "open", "high", "low", "close", "volume", "industry"] if c in history.columns]]
    history["time"] = pd.to_datetime(history["time"], errors="coerce").dt.normalize()
    history = history.dropna(subset=["time", "open", "high", "low", "close", "volume"])
    return history.sort_values("time").reset_index(drop=True)


def _build_namespace(df: pd.DataFrame) -> Dict[str, Any]:
    ns = {name: getattr(op, name) for name in dir(op) if not name.startswith("_")}
    ns.update({"df": df, "np": np, "pd": pd})
    for col in df.columns:
        ns[col] = df[col]
    return ns


def _eval_alpha_series(formula: str, df: pd.DataFrame) -> Optional[pd.Series]:
    try:
        ns = _build_namespace(df)
        exec(formula, ns)
        series = ns.get("alpha")
        if not isinstance(series, pd.Series):
            return None
        series = series.replace([np.inf, -np.inf], np.nan)
        return series
    except Exception:
        return None


def _signal_value_today(series: pd.Series) -> Optional[float]:
    if series is None or series.empty:
        return None

    mu = series.rolling(60, min_periods=20).mean()
    std = series.rolling(60, min_periods=20).std()
    z = (series - mu) / (std + 1e-9)
    if z.empty:
        return None

    latest = _safe_float(z.dropna().iloc[-1]) if not z.dropna().empty else None
    return latest


def compute_ticker_signal(
    ticker: str,
    top_alphas: Optional[List[Dict[str, Any]]] = None,
    market_data_dir: Path = MARKET_DATA_DIR,
) -> Dict[str, Any]:
    ticker = ticker.upper().strip()
    alphas = top_alphas if top_alphas is not None else load_alpha_definitions()

    base = {
        "enabled": False,
        "ticker": ticker,
        "side": "neutral",
        "signal_today": None,
        "score": None,
        "rank": None,
        "ic_oos": None,
        "sharpe_oos": None,
        "return_oos": None,
        "n_alphas": len(alphas),
        "top_alphas": alphas,
        "as_of": datetime.now().isoformat(),
    }

    if not alphas:
        base["error"] = "No alpha definitions in alpha_library.json"
        return base

    df = _load_ticker_frame(ticker, market_data_dir=market_data_dir)
    if df is None or df.empty:
        base["error"] = f"No market data for {ticker}"
        return base

    votes: List[Tuple[float, float]] = []
    used_alphas: List[Dict[str, Any]] = []

    for idx, alpha in enumerate(alphas):
        formula = alpha.get("formula") or ""
        ic_oos = _safe_float(alpha.get("ic_oos"))
        if not formula or ic_oos is None:
            continue

        series = _eval_alpha_series(formula, df)
        sig = _signal_value_today(series) if series is not None else None
        if sig is None:
            continue

        votes.append((max(ic_oos, 1e-6), sig))
        used_alphas.append({
            "rank": idx + 1,
            "id": alpha.get("id", ""),
            "ic_oos": ic_oos,
            "sharpe_oos": _safe_float(alpha.get("sharpe_oos")),
            "return_oos": _safe_float(alpha.get("return_oos")),
            "signal_today": sig,
        })

    if not votes:
        base["error"] = "No valid alpha signal for this ticker"
        return base

    weight_sum = sum(w for w, _ in votes)
    composite = float(sum(w * s for w, s in votes) / (weight_sum + 1e-9))

    avg_ic = float(np.mean([a["ic_oos"] for a in alphas if _safe_float(a.get("ic_oos")) is not None]))
    avg_sh = float(np.mean([v for v in [_safe_float(a.get("sharpe_oos")) for a in alphas] if v is not None])) if any(_safe_float(a.get("sharpe_oos")) is not None for a in alphas) else None
    avg_ret = float(np.mean([v for v in [_safe_float(a.get("return_oos")) for a in alphas] if v is not None])) if any(_safe_float(a.get("return_oos")) is not None for a in alphas) else None

    if composite >= LONG_THRESHOLD:
        side = "long"
    elif composite > SLIGHTLY_LONG_THRESHOLD:
        side = "slightly long"
    elif composite <= SHORT_THRESHOLD:
        side = "short"
    elif composite < SLIGHTLY_SHORT_THRESHOLD:
        side = "slightly short"
    else:
        side = "neutral"

    base.update({
        "enabled": True,
        "side": side,
        "signal_today": round(composite, 6),
        "score": round(composite, 6),
        "ic_oos": round(avg_ic, 6) if np.isfinite(avg_ic) else None,
        "sharpe_oos": round(avg_sh, 6) if avg_sh is not None and np.isfinite(avg_sh) else None,
        "return_oos": round(avg_ret, 6) if avg_ret is not None and np.isfinite(avg_ret) else None,
        "used_alphas": used_alphas,
    })
    return base


def _read_last_local_day(path: Path) -> Optional[pd.Timestamp]:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, usecols=["time"])
        if df.empty:
            return None
        s = pd.to_datetime(df["time"], errors="coerce").dropna()
        if s.empty:
            return None
        return s.max().normalize()
    except Exception:
        return None


def _should_refresh(ticker_path: Path, now: datetime, daily_start_hour: int) -> Tuple[bool, str]:
    """
    Quyết định có cần refresh hay không.
    Returns: (should_refresh, reason)
    """
    last_day = _read_last_local_day(ticker_path)
    today = now.date()

    if last_day is None:
        return True, "no_data"

    last_date = last_day.date()
    days_behind = (today - last_date).days

    # Dữ liệu đã là ngày hôm nay → skip
    if last_date >= today:
        return False, "up_to_date"

    # Lệch từ 2 ngày → cập nhật ngay bất kể giờ
    if days_behind >= 2:
        return True, f"backfill_{days_behind}d"

    # Lệch 1-2 ngày: kiểm tra giờ
    if now.hour >= daily_start_hour:
        return True, "scheduled_update"
    else:
        return False, "before_market_open"


def _refresh_ticker_market_data(
    ticker: str,
    market_data_dir: Path = MARKET_DATA_DIR,
    daily_start_hour: int = 9,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Best-effort incremental update + tính indicator cho một ticker."""
    if now is None:
        now = datetime.now()

    path = market_data_dir / f"{ticker}.csv"
    market_data_dir.mkdir(parents=True, exist_ok=True)

    last_day = _read_last_local_day(path)
    end_day = now.date()
    before_day = last_day.date() if last_day is not None else None

    should, reason = _should_refresh(path, now, daily_start_hour)
    if not should:
        return {
            "changed": False,
            "new_rows": 0,
            "before_day": before_day,
            "after_day": before_day,
            "attempted": False,
            "reason": reason,
        }

    if last_day is not None:
        start_date = (last_day.date() - timedelta(days=5)).strftime("%Y-%m-%d")
    else:
        start_date = (end_day - timedelta(days=365 * 5)).strftime("%Y-%m-%d")

    use_vnstock = ticker.upper() in VNSTOCK_FALLBACK_TICKERS
    if use_vnstock:
        remote = _fetch_ohlcv_vnstock(ticker, start=start_date, end=end_day.strftime("%Y-%m-%d"))
    else:
        remote = _fetch_ohlcv_yfinance(ticker, start=start_date, end=end_day.strftime("%Y-%m-%d"))

    if remote.empty:
        return {
            "changed": False,
            "new_rows": 0,
            "before_day": before_day,
            "after_day": before_day,
            "attempted": True,
            "reason": reason,
        }

    cols = [c for c in ["time", "ticker", "open", "high", "low", "close", "volume", "industry"] if c in remote.columns]
    if "time" not in cols:
        return {
            "changed": False,
            "new_rows": 0,
            "before_day": before_day,
            "after_day": before_day,
            "attempted": True,
            "reason": reason,
        }

    remote = remote[cols].copy()
    if "ticker" not in remote.columns:
        remote["ticker"] = ticker.upper()
    if "industry" not in remote.columns:
        remote["industry"] = pd.NA
    remote["time"] = pd.to_datetime(remote["time"], errors="coerce").dt.normalize()
    remote = remote.dropna(subset=["time", "open", "high", "low", "close", "volume"])

    if path.exists():
        local = pd.read_csv(path)
        local.columns = [c.lower() for c in local.columns]
        if "time" in local.columns:
            local["time"] = pd.to_datetime(local["time"], errors="coerce").dt.normalize()

        if "ticker" not in local.columns:
            local["ticker"] = ticker.upper()
        if "industry" not in local.columns:
            local["industry"] = pd.NA

        if local["industry"].notna().any():
            fallback_industry = local["industry"].dropna().iloc[-1]
            remote["industry"] = remote["industry"].fillna(fallback_industry)
        remote["ticker"] = remote["ticker"].fillna(ticker.upper())

        # Giữ các cột OHLCV + metadata từ local, bỏ indicator cũ để tính lại
        base_cols = ["time", "ticker", "open", "high", "low", "close", "volume", "industry"]
        local_base = local[[c for c in base_cols if c in local.columns]]
        remote_base = remote[[c for c in base_cols if c in remote.columns]]

        merged = pd.concat([local_base, remote_base], ignore_index=True)
    else:
        remote["ticker"] = remote["ticker"].fillna(ticker.upper())
        base_cols = ["time", "ticker", "open", "high", "low", "close", "volume", "industry"]
        merged = remote[[c for c in base_cols if c in remote.columns]]

    # Dedup: keep='last' để dữ liệu mới crawl ghi đè dữ liệu cũ cùng ngày
    merged = merged.drop_duplicates(subset=["time"], keep="last").sort_values("time")

    # Tính lại toàn bộ indicators trên dữ liệu đã merge
    merged = _recalculate_indicators(merged, ticker)

    before = 0
    if path.exists():
        try:
            before = len(pd.read_csv(path, usecols=["time"]))
        except Exception:
            before = 0

    merged.to_csv(path, index=False)
    after = len(merged)
    merged_last_ts = pd.to_datetime(merged["time"], errors="coerce").dropna().max()
    after_day = merged_last_ts.date() if pd.notna(merged_last_ts) else before_day

    return {
        "changed": after > before or reason.startswith("backfill"),
        "new_rows": max(after - before, 0),
        "before_day": before_day,
        "after_day": after_day,
        "attempted": True,
        "reason": reason,
    }


def _recalculate_indicators(merged: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Tính lại toàn bộ technical indicators và gộp vào DataFrame."""
    try:
        ohlcv = merged.set_index("time")[["open", "high", "low", "close", "volume"]].copy()
        ohlcv.index = pd.to_datetime(ohlcv.index)
        ohlcv = ohlcv.sort_index()

        df_with_indicators = add_technical_indicators(ohlcv)
        df_with_indicators = df_with_indicators.reset_index().rename(columns={"index": "time"})
        df_with_indicators["time"] = pd.to_datetime(df_with_indicators["time"]).dt.normalize()

        # Lấy lại metadata columns
        meta_cols = [c for c in ["ticker", "industry"] if c in merged.columns]
        if meta_cols:
            meta = merged[["time"] + meta_cols].copy()
            meta["time"] = pd.to_datetime(meta["time"]).dt.normalize()
            df_with_indicators = df_with_indicators.merge(meta, on="time", how="left")

        # Sắp xếp cột: time, ticker, OHLCV, industry, indicators
        first_cols = ["time", "ticker", "open", "high", "low", "close", "volume", "industry"]
        existing_first = [c for c in first_cols if c in df_with_indicators.columns]
        rest_cols = [c for c in df_with_indicators.columns if c not in first_cols]
        df_with_indicators = df_with_indicators[existing_first + rest_cols]

        return df_with_indicators
    except Exception as e:
        log.warning("[DailyRunner] Failed to calculate indicators for %s: %s", ticker, e)
        # Fallback: trả về merged không có indicators
        ordered_cols = ["time", "ticker", "open", "high", "low", "close", "volume", "industry"]
        for c in ordered_cols:
            if c not in merged.columns:
                merged[c] = pd.NA
        return merged[[c for c in ordered_cols if c in merged.columns]]


def refresh_market_data_daily(
    tickers: Optional[List[str]] = None,
    daily_start_hour: int = 9,
    skip_holiday_check: bool = False,
) -> Dict[str, Any]:
    """Incrementally update market CSVs for selected tickers."""
    if tickers is None:
        tickers = sorted([p.stem.upper() for p in MARKET_DATA_DIR.glob("*.csv")])
    total_tickers = len(tickers)

    now = datetime.now()
    changed, skipped = [], []
    failed = []
    new_rows_total = 0
    holiday_detected = False
    holiday_probe_ticker = None

    # Kiểm tra holiday: thử HOLIDAY_PROBE_COUNT mã đầu tiên
    probe_no_new = 0

    for idx, ticker in enumerate(tickers, start=1):
        try:
            result = _refresh_ticker_market_data(
                ticker,
                daily_start_hour=daily_start_hour,
                now=now,
            )
            is_changed = bool(result.get("changed"))
            n_new = int(result.get("new_rows") or 0)
            attempted = bool(result.get("attempted"))
            reason = result.get("reason", "")

            # Kiểm tra holiday từ các mã probe đầu tiên
            if idx <= HOLIDAY_PROBE_COUNT and attempted:
                if holiday_probe_ticker is None:
                    holiday_probe_ticker = ticker
                before_day = result.get("before_day")
                after_day = result.get("after_day")
                if before_day is not None and after_day is not None and after_day <= before_day:
                    probe_no_new += 1
                else:
                    # Có mã mới → không phải holiday
                    probe_no_new = 0

            if not skip_holiday_check and idx == HOLIDAY_PROBE_COUNT and probe_no_new >= HOLIDAY_PROBE_COUNT:
                holiday_detected = True
                log.info(
                    "[DailyRunner] Holiday detected: %d/%d probe tickers had no new data. Stopping.",
                    probe_no_new, HOLIDAY_PROBE_COUNT,
                )
                skipped.append(ticker)
                break

            if is_changed:
                changed.append(ticker)
                new_rows_total += n_new
            else:
                skipped.append(ticker)
        except Exception:
            log.exception("[DailyRunner] Error refreshing %s", ticker)
            failed.append(ticker)

    return {
        "target_day": date.today().isoformat(),
        "updated": changed,
        "skipped": skipped,
        "failed": failed,
        "new_rows": new_rows_total,
        "holiday_detected": holiday_detected,
        "holiday_probe_ticker": holiday_probe_ticker,
        "completed": len(changed) + len(skipped) + len(failed),
        "total": total_tickers,
    }


def build_daily_signals_snapshot(tickers: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
    if tickers is None:
        tickers = sorted([p.stem.upper() for p in MARKET_DATA_DIR.glob("*.csv")])
    all_alphas = load_alpha_definitions()
    snapshot: Dict[str, Dict[str, Any]] = {}
    for ticker in tickers:
        snapshot[ticker] = compute_ticker_signal(ticker, top_alphas=all_alphas)
    return snapshot


def save_signals_snapshot(snapshot: Dict[str, Dict[str, Any]]) -> Path:
    SIGNALS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for ticker, sig in snapshot.items():
        rows.append({
            "ticker":       ticker,
            "side":         sig.get("side", "neutral"),
            "signal_today": sig.get("signal_today"),
            "ic_oos":       sig.get("ic_oos"),
            "sharpe_oos":   sig.get("sharpe_oos"),
            "return_oos":   sig.get("return_oos"),
            "enabled":      sig.get("enabled", False),
            "as_of":        sig.get("as_of", ""),
        })
    pd.DataFrame(rows).to_csv(SIGNALS_PATH, index=False)
    return SIGNALS_PATH


def run_daily_update(
    tickers: Optional[List[str]] = None,
    daily_start_hour: int = 9,
    skip_holiday_check: bool = False,
) -> Dict[str, Any]:
    market = refresh_market_data_daily(tickers=tickers, daily_start_hour=daily_start_hour, skip_holiday_check=skip_holiday_check)

    if market.get("holiday_detected"):
        log.info("[DailyRunner] Holiday detected — skipping signal snapshot rebuild.")
        return {"market": market, "signals_file": None, "n_signals": 0}

    snapshot = build_daily_signals_snapshot(tickers=tickers)
    signal_path = save_signals_snapshot(snapshot)
    log.info("[DailyRunner] Daily runner hoàn thành %d/%d mã", market.get("completed", 0), market.get("total", 0))
    return {"market": market, "signals_file": str(signal_path), "n_signals": len(snapshot)}
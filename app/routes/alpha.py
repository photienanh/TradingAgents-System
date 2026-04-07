"""
app/routes/alpha.py
═══════════════════════════════════════════════════════════════════════
FastAPI router cho Alpha management — tích hợp AlphaGPT vào
TradingAgents API.

Endpoints:
  GET  /api/alpha/overview              — tổng quan tất cả tickers
  GET  /api/alpha/ticker/{ticker}       — chi tiết alpha của một mã
  GET  /api/alpha/signals               — top signals hôm nay
  GET  /api/alpha/decay                 — alpha decay monitor
  GET  /api/alpha/memory/stats          — RAG memory stats
  POST /api/alpha/gen/{ticker}          — trigger gen alpha (async)
  GET  /api/alpha/gen/status/{job_id}   — poll job status
  GET  /api/alpha/ticker/{ticker}/history — lịch sử refinement
  POST /api/alpha/daily-update          — refresh market + sentiment + alpha values
"""

import os
import json
import glob
import logging
import threading
import subprocess
import uuid
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import APIRouter
from fastapi.responses import JSONResponse

# ── Import từ alpha.core ──────────────────────────────────────────────
from alpha.core.universe import TICKER_INDUSTRY, VN30_SYMBOLS
from alpha.core.paths import (
    ALPHA_FORMULA_DIR, ALPHA_VALUES_DIR,
    SIGNALS_DIR, FEATURES_DIR,
)
from alpha.core.daily_runner import compute_composite

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/alpha", tags=["alpha"])

# ── Job store (in-memory, giống pattern trong app/main.py) ────────────
GEN_JOBS: dict[str, dict] = {}
GEN_JOBS_LOCK = threading.Lock()


# ── Helpers ───────────────────────────────────────────────────────────

def _load_meta(ticker: str) -> dict | None:
    path = os.path.join(ALPHA_FORMULA_DIR, f"{ticker}_alphas.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_alpha_values(ticker: str) -> pd.DataFrame | None:
    path = os.path.join(ALPHA_VALUES_DIR, f"{ticker}_alpha_values.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, index_col="time", parse_dates=True)
    df.index = pd.to_datetime(df.index).normalize()
    return df.replace([np.inf, -np.inf], np.nan)


def _load_close(ticker: str) -> pd.Series | None:
    path = os.path.join(FEATURES_DIR, f"{ticker}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=["time"])
    df["time"] = pd.to_datetime(df["time"]).dt.normalize()
    return df.set_index("time")["close"].sort_index()


def _safe_float(v):
    if v is None:
        return None
    try:
        fv = float(v)
        return fv if np.isfinite(fv) else None
    except Exception:
        return None


def _finite_mean(values) -> float | None:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size > 0 else None


# ── Routes ────────────────────────────────────────────────────────────

@router.get("/overview")
async def alpha_overview():
    """Tổng quan tất cả VN30 tickers — signal, IC, action."""
    rows = []
    available = []

    for ticker in VN30_SYMBOLS:
        meta = _load_meta(ticker)
        if meta is None:
            rows.append({
                "ticker": ticker,
                "industry": TICKER_INDUSTRY.get(ticker, "Khác"),
                "status": "no_data", "signal": None, "action": "—",
                "strength": 0, "avg_ic_oos": None, "avg_sharpe_oos": None,
                "n_alphas": 0,
            })
            continue

        available.append(ticker)
        ok = [a for a in meta["alphas"] if a.get("status") == "OK"]
        avg_ic = _finite_mean(abs(a.get("ic_oos") or a.get("ic") or 0) for a in ok)
        avg_sh = _finite_mean(a.get("sharpe_oos") or a.get("sharpe") or 0 for a in ok)

        av = _load_alpha_values(ticker)
        signal_val, action, strength = None, "HOLD", 0
        if av is not None and not av.empty:
            comp = compute_composite(ticker, av)
            if not comp.empty:
                latest = float(comp.iloc[-1])
                if np.isfinite(latest):
                    signal_val = round(latest, 4)
                    action = "BUY" if latest > 1.0 else ("SELL" if latest < -1.0 else "HOLD")
                    strength = min(100, int(abs(latest) * 40))

        rows.append({
            "ticker":         ticker,
            "industry":       TICKER_INDUSTRY.get(ticker, "Khác"),
            "status":         "ok",
            "signal":         signal_val,
            "action":         action,
            "strength":       strength,
            "avg_ic_oos":     round(float(avg_ic), 4) if avg_ic is not None else None,
            "avg_sharpe_oos": round(float(avg_sh), 4) if avg_sh  is not None else None,
            "n_alphas":       len(ok),
        })

    rows.sort(key=lambda r: r["signal"] or 0, reverse=True)
    for i, r in enumerate(rows):
        r["rank"] = i + 1

    return {
        "tickers":     rows,
        "n_available": len(available),
        "n_total":     len(VN30_SYMBOLS),
        "updated_at":  datetime.now().isoformat(),
    }


@router.get("/ticker/{ticker}")
async def alpha_ticker(ticker: str):
    """Chi tiết alpha, rolling IC, composite signal cho một mã."""
    ticker = ticker.upper()
    meta = _load_meta(ticker)
    if meta is None:
        return JSONResponse({"error": f"No alpha data for {ticker}"}, status_code=404)

    av    = _load_alpha_values(ticker)
    close = _load_close(ticker)

    alphas_out = []
    for a in meta["alphas"]:
        alphas_out.append({
            "id":         a["id"],
            "family":     a.get("family", "?"),
            "idea":       a.get("idea", ""),
            "hypothesis": a.get("hypothesis", ""),
            "expression": a.get("expression", ""),
            "ic":         _safe_float(a.get("ic")),
            "ic_oos":     _safe_float(a.get("ic_oos")),
            "ic_5d":      _safe_float(a.get("ic_5d")),
            "sharpe":     _safe_float(a.get("sharpe")),
            "sharpe_oos": _safe_float(a.get("sharpe_oos")),
            "turnover":   _safe_float(a.get("turnover")),
            "score":      _safe_float(a.get("score")),
            "status":     a.get("status", "?"),
            "flipped":    a.get("flipped", False),
            "gp_enhanced":a.get("gp_enhanced", False),
        })

    # Rolling IC
    rolling_ic_data = []
    if av is not None:
        feat_path = os.path.join(FEATURES_DIR, f"{ticker}.csv")
        if os.path.exists(feat_path):
            df_feat = pd.read_csv(feat_path, parse_dates=["time"])
            df_feat["time"] = pd.to_datetime(df_feat["time"]).dt.normalize()
            df_feat = df_feat.set_index("time").sort_index()
            fwd_ret = df_feat["close"].pct_change(1).shift(-1)
            for a in meta["alphas"]:
                if a.get("status") != "OK":
                    continue
                col = f"alpha_{a['id']}"
                if col not in av.columns:
                    continue
                merged = pd.concat([av[col], fwd_ret], axis=1).dropna()
                merged.columns = ["a", "r"]
                merged["a_rank"] = merged["a"].rolling(20).rank()
                merged["r_rank"] = merged["r"].rolling(20).rank()
                rolling = (
                    merged["a_rank"].rolling(20).corr(merged["r_rank"])
                    .dropna().tail(200)
                )
                rolling_ic_data.append({
                    "alpha_id": a["id"],
                    "family":   a.get("family", "?"),
                    "dates":    [d.strftime("%Y-%m-%d") for d in rolling.index],
                    "values":   [round(float(v), 4) for v in rolling.values if np.isfinite(v)],
                })

    # Composite
    composite_data = []
    if av is not None:
        comp = compute_composite(ticker, av).tail(200)
        composite_data = [
            {"date": d.strftime("%Y-%m-%d"), "value": round(float(v), 4)}
            for d, v in comp.items() if np.isfinite(v)
        ]

    # Close price
    close_data = []
    if close is not None:
        for d, v in close.tail(200).items():
            if np.isfinite(v):
                close_data.append({"date": d.strftime("%Y-%m-%d"), "value": round(float(v), 2)})

    latest_signal = composite_data[-1]["value"] if composite_data else 0.0
    action = "BUY" if latest_signal > 1.0 else ("SELL" if latest_signal < -1.0 else "HOLD")

    return {
        "ticker":        ticker,
        "industry":      TICKER_INDUSTRY.get(ticker, "Khác"),
        "n_rounds":      meta.get("n_rounds", 1),
        "n_rows":        meta.get("n_rows", 0),
        "alphas":        alphas_out,
        "rolling_ic":    rolling_ic_data,
        "composite":     composite_data,
        "close":         close_data,
        "latest_signal": latest_signal,
        "action":        action,
        "strength":      min(100, int(abs(latest_signal) * 40)),
    }


@router.get("/signals")
async def alpha_signals():
    """Top BUY/SELL signals hôm nay."""
    rows = []
    for ticker in VN30_SYMBOLS:
        av = _load_alpha_values(ticker)
        if av is None or av.empty:
            continue
        comp = compute_composite(ticker, av)
        if comp.empty:
            continue
        latest = float(comp.iloc[-1])
        if not np.isfinite(latest):
            continue
        close = _load_close(ticker)
        price = float(close.iloc[-1]) if close is not None and not close.empty else None
        rows.append({
            "ticker":   ticker,
            "industry": TICKER_INDUSTRY.get(ticker, "Khác"),
            "signal":   round(latest, 4),
            "action":   "BUY" if latest > 1.0 else ("SELL" if latest < -1.0 else "HOLD"),
            "strength": min(100, int(abs(latest) * 40)),
            "price":    price,
        })

    rows.sort(key=lambda r: r["signal"], reverse=True)
    for i, r in enumerate(rows):
        r["rank"] = i + 1

    return {
        "all":      rows,
        "top_buy":  [r for r in rows if r["action"] == "BUY"][:5],
        "top_sell": [r for r in rows if r["action"] == "SELL"][:5],
        "date":     datetime.now().strftime("%Y-%m-%d"),
    }


@router.get("/decay")
async def alpha_decay():
    """Phát hiện alpha đang suy giảm IC."""
    from alpha.core.backtester import detect_decay
    results = []
    for ticker in VN30_SYMBOLS:
        meta = _load_meta(ticker)
        av   = _load_alpha_values(ticker)
        if meta is None or av is None:
            continue
        feat_path = os.path.join(FEATURES_DIR, f"{ticker}.csv")
        if not os.path.exists(feat_path):
            continue
        try:
            df = pd.read_csv(feat_path, parse_dates=["time"])
            df["time"] = pd.to_datetime(df["time"]).dt.normalize()
            df = df.set_index("time").sort_index()
            fwd_ret = df["close"].pct_change(1).shift(-1).replace([np.inf, -np.inf], np.nan)
            decaying = []
            for a in meta["alphas"]:
                if a.get("status") != "OK":
                    continue
                col = f"alpha_{a['id']}"
                if col not in av.columns:
                    continue
                d = detect_decay(av[col].replace([np.inf, -np.inf], np.nan), fwd_ret)
                if d.get("decaying"):
                    decaying.append({"alpha_id": a["id"], **d})
            if decaying:
                results.append({"ticker": ticker, "decaying": decaying})
        except Exception as e:
            log.error(f"[{ticker}] decay error: {e}")

    return {"needs_refinement": results, "count": len(results)}


@router.get("/memory/stats")
async def alpha_memory_stats():
    """Thống kê RAG memory."""
    from alpha.core.alpha_memory import AlphaMemory
    from alpha.core.paths import ALPHA_MEMORY_DIR
    mem = AlphaMemory(ALPHA_MEMORY_DIR)
    stats = {"global": mem.stats(), "tickers": {}}
    for ticker in VN30_SYMBOLS:
        s = mem.stats(ticker)
        if s["ticker_count"] > 0:
            stats["tickers"][ticker] = s
    return stats


@router.post("/gen/{ticker}")
async def alpha_gen(ticker: str, body: dict = {}):
    """
    Trigger alpha generation cho một ticker — chạy async background.
    Body: { "force": bool, "refine_only": bool, "no_gp": bool }
    """
    ticker = ticker.upper()
    if ticker not in VN30_SYMBOLS:
        return JSONResponse({"error": "Unknown ticker"}, status_code=400)

    force       = body.get("force", False)
    refine_only = body.get("refine_only", False)
    no_gp       = body.get("no_gp", False)

    job_id = str(uuid.uuid4())
    with GEN_JOBS_LOCK:
        GEN_JOBS[job_id] = {
            "job_id": job_id, "ticker": ticker, "status": "running",
            "started_at": datetime.now().isoformat(),
            "finished_at": None, "return_code": None, "error": None,
        }

    def _run():
        # Chạy như subprocess để tránh blocking event loop
        cmd = [sys.executable, "-m", "alpha.pipelines.gen_alpha", "--ticker", ticker]
        if refine_only:
            cmd.append("--refine-only")
        elif force:
            cmd.append("--force")
        if no_gp:
            cmd.append("--no-gp")

        try:
            # Chạy từ root TradingAgents directory
            project_root = str(Path(__file__).resolve().parents[2])
            completed = subprocess.run(cmd, cwd=project_root, check=False)
            with GEN_JOBS_LOCK:
                GEN_JOBS[job_id]["return_code"] = completed.returncode
                GEN_JOBS[job_id]["finished_at"] = datetime.now().isoformat()
                GEN_JOBS[job_id]["status"] = (
                    "completed" if completed.returncode == 0 else "failed"
                )
                if completed.returncode != 0:
                    GEN_JOBS[job_id]["error"] = f"Exit code {completed.returncode}"
        except Exception as e:
            with GEN_JOBS_LOCK:
                GEN_JOBS[job_id]["status"]      = "failed"
                GEN_JOBS[job_id]["error"]       = str(e)
                GEN_JOBS[job_id]["finished_at"] = datetime.now().isoformat()

    threading.Thread(target=_run, daemon=True).start()
    return {"status": "started", "job_id": job_id, "ticker": ticker}


@router.get("/gen/status/{job_id}")
async def alpha_gen_status(job_id: str):
    """Poll trạng thái của một gen job."""
    with GEN_JOBS_LOCK:
        job = GEN_JOBS.get(job_id)
    if job is None:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    return job


@router.get("/ticker/{ticker}/history")
async def alpha_history(ticker: str):
    """Lịch sử refinement của một ticker."""
    ticker = ticker.upper()
    meta = _load_meta(ticker)
    if meta is None:
        return JSONResponse({"error": "No data"}, status_code=404)
    return {"history": meta.get("history", [])}


@router.post("/daily-update")
async def alpha_daily_update(body: dict = {}):
    """
    Trigger daily update pipeline:
    refresh market data → refresh sentiment → recompute alpha values.
    Tương đương chạy daily_runner.py --all
    """
    tickers = body.get("tickers", VN30_SYMBOLS)
    skip_market    = body.get("skip_market", False)
    skip_sentiment = body.get("skip_sentiment", False)
    skip_alpha     = body.get("skip_alpha", False)

    job_id = str(uuid.uuid4())
    with GEN_JOBS_LOCK:
        GEN_JOBS[job_id] = {
            "job_id": job_id, "type": "daily_update",
            "tickers": tickers, "status": "running",
            "started_at": datetime.now().isoformat(),
        }

    def _run():
        try:
            from alpha.core.daily_runner import (
                refresh_latest_market_data,
                refresh_latest_sentiment_data,
                refresh_alpha_values_from_existing_formulas,
            )
            result = {}
            if not skip_market:
                result["market"] = refresh_latest_market_data(tickers)
            if not skip_sentiment:
                target_day = None
                if result.get("market", {}).get("target_day"):
                    target_day = pd.Timestamp(result["market"]["target_day"]).normalize()
                result["sentiment"] = refresh_latest_sentiment_data(tickers, target_day)
            if not skip_alpha:
                result["alpha"] = refresh_alpha_values_from_existing_formulas(tickers)

            with GEN_JOBS_LOCK:
                GEN_JOBS[job_id].update({
                    "status": "completed",
                    "finished_at": datetime.now().isoformat(),
                    "result": {
                        k: {kk: len(vv) if isinstance(vv, list) else vv
                            for kk, vv in v.items()}
                        for k, v in result.items()
                    },
                })
        except Exception as e:
            with GEN_JOBS_LOCK:
                GEN_JOBS[job_id].update({
                    "status": "failed",
                    "error": str(e),
                    "finished_at": datetime.now().isoformat(),
                })

    threading.Thread(target=_run, daemon=True).start()
    return {"status": "started", "job_id": job_id}
"""FastAPI endpoints for AlphaGPT daily signal integration."""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from alpha.daily_runner import MARKET_DATA_DIR, get_top_alphas_by_ic_oos
from alpha.manager import (
    get_all_signals,
    get_signal_for_ticker,
    trigger_daily_update_async,
)

router = APIRouter(prefix="/api/alpha", tags=["alpha"])


def _as_action(side: str) -> str:
    if side == "long":
        return "BUY"
    if side == "short":
        return "SELL"
    return "HOLD"


@router.get("/overview")
async def alpha_overview() -> Dict[str, Any]:
    tickers = sorted([p.stem.upper() for p in Path(MARKET_DATA_DIR).glob("*.csv")])
    top_alphas = get_top_alphas_by_ic_oos(limit=5)

    rows: List[Dict[str, Any]] = []
    for ticker in tickers:
        sig = get_signal_for_ticker(ticker)
        signal_today = sig.get("signal_today")
        rows.append(
            {
                "ticker": ticker,
                "signal_today": signal_today,
                "side": sig.get("side", "neutral"),
                "action": _as_action(sig.get("side", "neutral")),
                "ic_oos": sig.get("ic_oos"),
                "sharpe_oos": sig.get("sharpe_oos"),
                "return_oos": sig.get("return_oos"),
                "n_alphas": len(top_alphas),
                "enabled": sig.get("enabled", False),
            }
        )

    rows.sort(
        key=lambda x: abs(float(x.get("signal_today") or 0.0)),
        reverse=True,
    )
    for idx, row in enumerate(rows, start=1):
        row["rank"] = idx

    return {
        "updated_at": datetime.now().isoformat(),
        "n_tickers": len(rows),
        "top5_alpha_ids": [a.get("id", "") for a in top_alphas],
        "tickers": rows,
    }


@router.get("/signals")
async def alpha_signals() -> Dict[str, Any]:
    all_signals = get_all_signals()
    rows = []
    for ticker, sig in all_signals.items():
        side = sig.get("side", "neutral")
        rows.append(
            {
                "ticker": ticker,
                "signal_today": sig.get("signal_today"),
                "side": side,
                "action": _as_action(side),
                "ic_oos": sig.get("ic_oos"),
                "sharpe_oos": sig.get("sharpe_oos"),
                "return_oos": sig.get("return_oos"),
            }
        )

    rows.sort(
        key=lambda x: abs(float(x.get("signal_today") or 0.0)),
        reverse=True,
    )
    return {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "all": rows,
        "top_buy": [r for r in rows if r["action"] == "BUY"][:5],
        "top_sell": [r for r in rows if r["action"] == "SELL"][:5],
    }


@router.get("/ticker/{ticker}")
async def alpha_ticker(ticker: str) -> Dict[str, Any]:
    ticker = ticker.upper().strip()
    sig = get_signal_for_ticker(ticker)
    if not sig.get("enabled") and sig.get("error"):
        return JSONResponse({"error": sig["error"], "ticker": ticker}, status_code=404)

    return {
        "ticker": ticker,
        "signal_today": sig.get("signal_today"),
        "side": sig.get("side", "neutral"),
        "action": _as_action(sig.get("side", "neutral")),
        "ic_oos": sig.get("ic_oos"),
        "sharpe_oos": sig.get("sharpe_oos"),
        "return_oos": sig.get("return_oos"),
        "top_alphas": sig.get("top_alphas", []),
        "used_alphas": sig.get("used_alphas", []),
        "updated_at": sig.get("as_of"),
    }


@router.post("/daily-update")
async def alpha_daily_update(body: Dict[str, Any] | None = None) -> Dict[str, Any]:
    force = bool((body or {}).get("force", False))
    res = trigger_daily_update_async(force=force)
    return {
        "status": "started" if res.get("accepted") else "skipped",
        **res,
    }
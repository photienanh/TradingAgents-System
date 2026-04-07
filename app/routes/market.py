import datetime
import math
from typing import Optional

from fastapi import APIRouter
from fastapi.responses import JSONResponse

try:
    from vnstock import Listing, Trading
    VNSTOCK_AVAILABLE = True
except ImportError:
    VNSTOCK_AVAILABLE = False


router = APIRouter()


def _json_safe(value):
    """Convert non-finite floats recursively so JSON serialization never fails."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def _to_float_or_zero(value) -> float:
    try:
        v = float(value)
        return v if math.isfinite(v) else 0.0
    except (TypeError, ValueError):
        return 0.0

# Cache for stock market data (separate cache for each group)
stock_data_cache = {
    "cache_duration": 60,
    "all": {"data": None, "timestamp": None},
    "VN30": {"data": None, "timestamp": None},
    "HNX30": {"data": None, "timestamp": None},
    "HOSE": {"data": None, "timestamp": None},
    "HNX": {"data": None, "timestamp": None},
    "UPCOM": {"data": None, "timestamp": None},
}


@router.get("/api/market/data")
async def get_market_data(group: Optional[str] = None):
    """Get stock market data from vnstock."""
    if not VNSTOCK_AVAILABLE:
        return JSONResponse(
            status_code=503,
            content={"error": "vnstock is not available. Please install it with: pip install vnstock"},
        )

    try:
        cache_key = group or "all"
        now = datetime.datetime.now()

        if cache_key in stock_data_cache:
            cached = stock_data_cache[cache_key]
            if cached["data"] is not None and cached["timestamp"] is not None:
                time_diff = (now - cached["timestamp"]).total_seconds()
                if time_diff < stock_data_cache["cache_duration"]:
                    return {"data": cached["data"], "cached": True, "group": cache_key}

        listing = Listing()
        trading = Trading()

        if group:
            group_upper = group.upper()
            if group_upper == "HOSE":
                symbols_df = listing.symbols_by_exchange("HOSE")
                symbols = list(symbols_df["symbol"]) if hasattr(symbols_df, "columns") else list(symbols_df)
            elif group_upper == "HNX":
                symbols_df = listing.symbols_by_exchange("HNX")
                symbols = list(symbols_df["symbol"]) if hasattr(symbols_df, "columns") else list(symbols_df)
            elif group_upper == "UPCOM":
                symbols_df = listing.symbols_by_exchange("UPCOM")
                symbols = list(symbols_df["symbol"]) if hasattr(symbols_df, "columns") else list(symbols_df)
            elif group_upper == "VN30":
                symbols = list(listing.symbols_by_group("VN30"))
            elif group_upper == "HNX30":
                symbols = list(listing.symbols_by_group("HNX30"))
            else:
                symbols = list(listing.all_symbols()["symbol"])
        else:
            symbols = list(listing.all_symbols()["symbol"])

        price_data = trading.price_board(symbols)
        result = price_data.to_dict("records")
        result = _json_safe(result)
        result = [
            r for r in result
            if _to_float_or_zero(r.get("close_price")) > 0
            or _to_float_or_zero(r.get("reference_price")) > 0
        ]
        result = sorted(result, key=lambda x: x.get("symbol", ""))

        if cache_key not in stock_data_cache:
            stock_data_cache[cache_key] = {"data": None, "timestamp": None}

        stock_data_cache[cache_key]["data"] = result
        stock_data_cache[cache_key]["timestamp"] = now

        return {"data": result, "cached": False, "group": cache_key, "count": len(result)}

    except Exception as exc:
        print(f"Error fetching market data: {exc}")
        return JSONResponse(status_code=500, content={"error": str(exc)})


@router.get("/api/market/symbols")
async def get_symbols():
    """Get all available stock symbols."""
    if not VNSTOCK_AVAILABLE:
        return JSONResponse(status_code=503, content={"error": "vnstock is not available"})

    try:
        listing = Listing()
        symbols_df = listing.all_symbols()
        symbols = symbols_df.to_dict("records")
        return {"symbols": _json_safe(symbols)}
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})


@router.get("/api/market/indices")
async def get_market_indices():
    """Get market indices (VN-INDEX, HNX-INDEX, UPCOM-INDEX)."""
    if not VNSTOCK_AVAILABLE:
        return JSONResponse(status_code=503, content={"error": "vnstock is not available"})

    try:
        from datetime import datetime, timedelta
        from vnstock import Quote

        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")

        indices = {}

        try:
            q = Quote(symbol="VNINDEX")
            df = q.history(start=start_date, end=end_date)
            if not df.empty and len(df) >= 2:
                latest = df.iloc[-1]
                prev = df.iloc[-2]
                indices["VNINDEX"] = {
                    "value": float(latest["close"]),
                    "change": float(latest["close"] - prev["close"]),
                    "percent": float((latest["close"] - prev["close"]) / prev["close"] * 100),
                }
            else:
                indices["VNINDEX"] = {"value": 0, "change": 0, "percent": 0}
        except Exception as exc:
            print(f"Error fetching VNINDEX: {exc}")
            indices["VNINDEX"] = {"value": 0, "change": 0, "percent": 0}

        try:
            q = Quote(symbol="HNXINDEX")
            df = q.history(start=start_date, end=end_date)
            if not df.empty and len(df) >= 2:
                latest = df.iloc[-1]
                prev = df.iloc[-2]
                indices["HNX"] = {
                    "value": float(latest["close"]),
                    "change": float(latest["close"] - prev["close"]),
                    "percent": float((latest["close"] - prev["close"]) / prev["close"] * 100),
                }
            else:
                indices["HNX"] = {"value": 0, "change": 0, "percent": 0}
        except Exception as exc:
            print(f"Error fetching HNX: {exc}")
            indices["HNX"] = {"value": 0, "change": 0, "percent": 0}

        try:
            q = Quote(symbol="UPCOMINDEX")
            df = q.history(start=start_date, end=end_date)
            if not df.empty and len(df) >= 2:
                latest = df.iloc[-1]
                prev = df.iloc[-2]
                indices["UPCOM"] = {
                    "value": float(latest["close"]),
                    "change": float(latest["close"] - prev["close"]),
                    "percent": float((latest["close"] - prev["close"]) / prev["close"] * 100),
                }
            else:
                indices["UPCOM"] = {"value": 0, "change": 0, "percent": 0}
        except Exception as exc:
            print(f"Error fetching UPCOM: {exc}")
            indices["UPCOM"] = {"value": 0, "change": 0, "percent": 0}

        try:
            q = Quote(symbol="VN30")
            df = q.history(start=start_date, end=end_date)
            if not df.empty and len(df) >= 2:
                latest = df.iloc[-1]
                prev = df.iloc[-2]
                indices["VN30"] = {
                    "value": float(latest["close"]),
                    "change": float(latest["close"] - prev["close"]),
                    "percent": float((latest["close"] - prev["close"]) / prev["close"] * 100),
                }
            else:
                indices["VN30"] = {"value": 0, "change": 0, "percent": 0}
        except Exception as exc:
            print(f"Error fetching VN30: {exc}")
            indices["VN30"] = {"value": 0, "change": 0, "percent": 0}

        return {"indices": _json_safe(indices)}

    except Exception as exc:
        print(f"Error fetching indices: {exc}")
        return JSONResponse(status_code=500, content={"error": str(exc)})

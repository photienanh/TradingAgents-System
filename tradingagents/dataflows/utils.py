import pandas as pd
from datetime import timedelta, datetime
from typing import Optional

VN30_INDUSTRIES = {
    "Ngân hàng": ["ACB", "BID", "CTG", "HDB", "LPB", "MBB", "SHB", "SSB", "STB", "TCB", "TPB", "VCB", "VIB", "VPB"],
    "Bất động sản": ["VHM", "VIC", "VRE", "VPL"],
    "Bán lẻ": ["MWG"],
    "Công nghệ": ["FPT"],
    "Tiêu dùng": ["MSN", "SAB", "VNM"],
    "Thép": ["HPG"],
    "Dầu khí": ["GAS", "PLX"],
    "Hóa chất": ["DGC", "GVR"],
    "Hàng không": ["VJC"],
    "Chứng khoán": ["SSI"],
}

VN30_SYMBOLS = [
    "ACB", "BID", "CTG", "DGC", "FPT", "GAS",
    "GVR", "HDB", "HPG", "LPB", "MBB", "MSN",
    "MWG", "PLX", "SAB", "SHB", "SSB", "SSI",
    "STB", "TCB", "TPB", "VCB", "VHM", "VIB",
    "VIC", "VJC", "VNM", "VPB", "VPL", "VRE",
]

def build_date_window(
    curr_date: Optional[str],
    look_back_days: int,
    *,
    min_lookback_days: Optional[int] = None,
    end_of_day: bool = False,
) -> tuple[datetime, datetime, str, str]:
    """Build a normalized [start_dt, end_dt] window and YYYY-mm-dd strings."""
    if look_back_days < 0:
        raise ValueError("look_back_days phải >= 0")

    base_dt = datetime.strptime(curr_date, "%Y-%m-%d") if curr_date else datetime.now()
    if end_of_day:
        end_dt = base_dt.replace(hour=23, minute=59, second=59, microsecond=0)
    else:
        end_dt = base_dt

    effective_lookback = look_back_days
    if min_lookback_days is not None:
        effective_lookback = max(look_back_days, min_lookback_days)

    start_dt = end_dt - timedelta(days=effective_lookback)
    if end_of_day:
        start_dt = start_dt.replace(hour=0, minute=0, second=0, microsecond=0)

    return start_dt, end_dt, start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")


def classify_trend(df: pd.DataFrame) -> str:
    """Classify trend from close-price regression slope over the window."""
    if df.empty or len(df) < 2:
        return "không đủ dữ liệu"

    closes = df["close"].values.astype(float)
    n = len(closes)
    x = list(range(n))
    mean_x = (n - 1) / 2
    mean_y = sum(closes) / n

    num = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, closes))
    den = sum((xi - mean_x) ** 2 for xi in x)

    if den == 0:
        return "đi ngang"

    slope = num / den
    threshold = mean_y * 0.001
    if slope > threshold:
        return "tăng"
    if slope < -threshold:
        return "giảm"
    return "đi ngang"


def day_change(df: pd.DataFrame, ref_date: pd.Timestamp) -> dict:
    """Return open/close and daily percentage change for a reference date."""
    row = df[df["date"] == ref_date]
    if row.empty:
        return {
            "date": ref_date.date(),
            "status": "N/A (không phải ngày giao dịch)",
            "open": None,
            "close": None,
            "change_pct": None,
        }

    r = row.iloc[0]
    change_pct = (r["close"] - r["open"]) / r["open"] * 100 if r["open"] else None
    direction = "tăng" if (change_pct or 0) > 0 else ("giảm" if (change_pct or 0) < 0 else "đi ngang")
    return {
        "date": ref_date.date(),
        "open": round(float(r["open"]), 2),
        "close": round(float(r["close"]), 2),
        "change_pct": round(change_pct, 2) if change_pct is not None else None,
        "status": direction,
    }


def filter_window(df: pd.DataFrame, ref_dt: datetime, days: int) -> pd.DataFrame:
    """Filter rows where date is in [ref_dt - days, ref_dt]."""
    start_ts = pd.Timestamp(ref_dt - timedelta(days=days)).normalize()
    end_ts = pd.Timestamp(ref_dt).normalize()
    return df[(df["date"] >= start_ts) & (df["date"] <= end_ts)].reset_index(drop=True)


def format_trend_block(df_window: pd.DataFrame, label: str) -> str:
    """Render a compact trend summary line for one window."""
    if df_window.empty:
        return f"  {label}: không đủ dữ liệu"

    trend = classify_trend(df_window)
    return (
        f"  {label}: **{trend}** "
        f"(đóng cửa đầu kỳ: {df_window.iloc[0]['close']:,.2f} | "
        f"cuối kỳ: {df_window.iloc[-1]['close']:,.2f})"
    )

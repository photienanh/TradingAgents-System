from typing import Annotated, Optional
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import yfinance as yf
import pandas as pd
from .stockstats_utils import StockstatsUtils
from alpha.core.universe import VN30_SYMBOLS

def _normalize_vn_ticker(symbol: str) -> str:
    """Normalize ticker to Yahoo Finance VN format."""
    ticker = symbol.strip().upper()
    return ticker if ticker.endswith(".VN") else f"{ticker}.VN"


def _display_ticker(symbol: str) -> str:
    """Return display ticker without Yahoo VN suffix."""
    ticker = symbol.strip().upper()
    return ticker[:-3] if ticker.endswith(".VN") else ticker

def get_YFin_data_online(
    symbol: Annotated[str, "ticker symbol of the company"],
    curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
    look_back_days: Annotated[int, "Number of days to look back"],
):
    if look_back_days < 0:
        return "look_back_days phải >= 0"

    end_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    start_dt = end_dt - relativedelta(days=look_back_days)
    start_date = start_dt.strftime("%Y-%m-%d")
    end_date = end_dt.strftime("%Y-%m-%d")

    # Create ticker object
    ticker = yf.Ticker(_normalize_vn_ticker(symbol))

    end_date_yf = (
        datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
    ).strftime("%Y-%m-%d")
    
    # Fetch historical data for the specified date range
    data = ticker.history(start=start_date, end=end_date_yf)

    # Check if data is empty
    if data.empty:
        return (
            f"Không có dữ liệu cho mã '{symbol}' từ {start_date} đến {end_date}"
        )

    # Remove timezone info from index for cleaner output
    if isinstance(data.index, pd.DatetimeIndex) and data.index.tz is not None:
        data.index = data.index.tz_localize(None)

    data.drop(columns=["Dividends", "Stock Splits"], inplace=True, errors="ignore")

    # Convert DataFrame to CSV string
    csv_string = data.to_csv()

    # Add header information
    header = f"# Dữ liệu cổ phiếu cho {symbol.upper()} từ {start_date} đến {end_date}\n"
    header += f"# Tổng số bản ghi: {len(data)}\n"
    header += f"# Dữ liệu được lấy vào lúc: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    return header + csv_string

def get_indicators(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicator: Annotated[str, "technical indicator to get the analysis and report of"],
    curr_date: Annotated[
        str, "The current trading date you are trading on, YYYY-mm-dd"
    ],
    look_back_days: Annotated[int, "how many days to look back"],
) -> str:

    best_ind_params = {
        # Moving Averages
        "close_50_sma": (
            "50 SMA: Chỉ báo xu hướng trung hạn. "
            "Cách dùng: Xác định hướng xu hướng và làm vùng hỗ trợ/kháng cự động. "
            "Lưu ý: Có độ trễ so với giá, nên kết hợp thêm chỉ báo nhanh để vào lệnh kịp thời."
        ),
        "close_200_sma": (
            "200 SMA: Mốc tham chiếu xu hướng dài hạn. "
            "Cách dùng: Xác nhận xu hướng tổng thể và nhận diện mô hình golden/death cross. "
            "Lưu ý: Phản ứng chậm, phù hợp xác nhận chiến lược hơn là tín hiệu vào lệnh thường xuyên."
        ),
        "close_10_ema": (
            "10 EMA: Đường trung bình ngắn hạn phản ứng nhanh. "
            "Cách dùng: Bắt các thay đổi động lượng sớm và tìm điểm vào lệnh tiềm năng. "
            "Lưu ý: Dễ nhiễu khi thị trường đi ngang, nên kết hợp với MA dài hơn để lọc tín hiệu giả."
        ),
        # MACD Related
        "macd": (
            "MACD: Đo động lượng thông qua chênh lệch giữa các EMA. "
            "Cách dùng: Theo dõi giao cắt và phân kỳ để nhận diện khả năng đổi xu hướng. "
            "Lưu ý: Nên xác nhận thêm bằng chỉ báo khác khi thị trường biến động thấp hoặc đi ngang."
        ),
        "macds": (
            "MACD Signal: Đường EMA làm mượt của MACD. "
            "Cách dùng: Dùng giao cắt giữa MACD và Signal để kích hoạt tín hiệu giao dịch. "
            "Lưu ý: Nên nằm trong chiến lược tổng thể để giảm tín hiệu sai."
        ),
        "macdh": (
            "MACD Histogram: Thể hiện độ lệch giữa MACD và đường Signal. "
            "Cách dùng: Quan sát độ mạnh động lượng và phát hiện phân kỳ sớm. "
            "Lưu ý: Có thể biến động mạnh, nên bổ sung bộ lọc khi thị trường chạy nhanh."
        ),
        # Momentum Indicators
        "rsi": (
            "RSI: Đo động lượng để nhận biết vùng quá mua/quá bán. "
            "Cách dùng: Dùng ngưỡng 70/30 và theo dõi phân kỳ để tìm tín hiệu đảo chiều. "
            "Lưu ý: Trong xu hướng mạnh RSI có thể neo ở vùng cực trị, cần đối chiếu thêm phân tích xu hướng."
        ),
        # Volatility Indicators
        "boll": (
            "Bollinger Middle: SMA 20 làm đường cơ sở của Bollinger Bands. "
            "Cách dùng: Là mốc động để đánh giá dao động giá. "
            "Lưu ý: Kết hợp với dải trên/dưới để nhận diện breakout hoặc đảo chiều hiệu quả hơn."
        ),
        "boll_ub": (
            "Bollinger Upper Band: Thường nằm trên đường giữa khoảng 2 độ lệch chuẩn. "
            "Cách dùng: Gợi ý vùng quá mua và vùng có thể bứt phá. "
            "Lưu ý: Cần xác nhận bằng công cụ khác; trong xu hướng mạnh giá có thể bám dải trên lâu."
        ),
        "boll_lb": (
            "Bollinger Lower Band: Thường nằm dưới đường giữa khoảng 2 độ lệch chuẩn. "
            "Cách dùng: Gợi ý vùng quá bán tiềm năng. "
            "Lưu ý: Nên có phân tích bổ sung để tránh bắt nhầm tín hiệu đảo chiều giả."
        ),
        "atr": (
            "ATR: Đo biến động bằng trung bình True Range. "
            "Cách dùng: Đặt stop-loss và điều chỉnh khối lượng vị thế theo mức biến động hiện tại. "
            "Lưu ý: Đây là chỉ báo phản ứng, nên dùng trong chiến lược quản trị rủi ro tổng thể."
        ),
        # Volume-Based Indicators
        "vwma": (
            "VWMA: Đường trung bình động có trọng số theo khối lượng. "
            "Cách dùng: Xác nhận xu hướng bằng cách kết hợp biến động giá với dữ liệu khối lượng. "
            "Lưu ý: Cẩn trọng khi có đột biến volume vì có thể làm lệch kết quả; nên kết hợp thêm chỉ báo volume khác."
        ),
        "mfi": (
            "MFI: Money Flow Index là chỉ báo động lượng dùng cả giá và khối lượng để đo áp lực mua bán. "
            "Cách dùng: Xác định quá mua (>80) hoặc quá bán (<20), đồng thời xác nhận độ mạnh của xu hướng hoặc khả năng đảo chiều. "
            "Lưu ý: Nên dùng cùng RSI hoặc MACD để xác nhận; phân kỳ giữa giá và MFI có thể báo hiệu đảo chiều."
        ),
    }

    if indicator not in best_ind_params:
        raise ValueError(
            f"Chỉ báo {indicator} chưa được hỗ trợ. Vui lòng chọn một trong các chỉ báo: {list(best_ind_params.keys())}"
        )

    end_date = curr_date
    curr_date_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    before = curr_date_dt - relativedelta(days=look_back_days)

    # Optimized: Get stock data once and calculate indicators for all dates
    try:
        indicator_data = _get_stock_stats_bulk(symbol, indicator)
        
        # Generate the date range we need
        current_dt = curr_date_dt
        date_values = []
        
        while current_dt >= before:
            date_str = current_dt.strftime('%Y-%m-%d')
            
            # Look up the indicator value for this date
            if date_str in indicator_data:
                indicator_value = indicator_data[date_str]
            else:
                indicator_value = "N/A: Không phải ngày giao dịch (cuối tuần hoặc ngày lễ)"
            
            date_values.append((date_str, indicator_value))
            current_dt = current_dt - relativedelta(days=1)
        
        # Build the result string
        ind_string = ""
        for date_str, value in date_values:
            ind_string += f"{date_str}: {value}\n"
        
    except Exception as e:
        print(f"Lỗi khi lấy dữ liệu chỉ báo theo dạng bulk từ yfinance: {e}")
        # Fallback to original implementation if bulk method fails
        ind_string = ""
        curr_date_dt = datetime.strptime(curr_date, "%Y-%m-%d")
        while curr_date_dt >= before:
            indicator_value = get_stockstats_indicator(
                symbol, indicator, curr_date_dt.strftime("%Y-%m-%d")
            )
            ind_string += f"{curr_date_dt.strftime('%Y-%m-%d')}: {indicator_value}\n"
            curr_date_dt = curr_date_dt - relativedelta(days=1)

    result_str = (
        f"## Giá trị {indicator} từ {before.strftime('%Y-%m-%d')} đến {end_date}:\n\n"
        + ind_string
        + "\n\n"
        + best_ind_params.get(indicator, "Không có mô tả cho chỉ báo này.")
    )

    return result_str

def _fetch_ohlcv_single(symbol: str, start: str, end: str) -> pd.DataFrame:
    """
    Lấy OHLCV cho một mã qua yfinance.
    end được cộng thêm 1 ngày vì yfinance dùng khoảng nửa mở [start, end).
    Trả về DataFrame trống nếu lỗi.
    """
    try:
        end_yf = (datetime.strptime(end, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
        df = yf.download(
            _normalize_vn_ticker(symbol),
            start=start,
            end=end_yf,
            multi_level_index=False,
            progress=False,
            auto_adjust=True,
        )
        if df is None or df.empty:
            return pd.DataFrame()
 
        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]          # → date, open, high, low, close, volume
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df = df.sort_values("date").reset_index(drop=True)
        return df
    except Exception as e:
        print(f"[get_market_context] Lỗi fetch {symbol}: {e}")
        return pd.DataFrame()
 
 
def _fetch_ohlcv_bulk(symbols: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    """
    Lấy OHLCV cho nhiều mã cùng lúc bằng yf.download bulk (1 request).
    Trả về dict {symbol: DataFrame}.
    """
    end_yf = (datetime.strptime(end, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    yf_symbols = [_normalize_vn_ticker(s) for s in symbols]
 
    try:
        raw = yf.download(
            yf_symbols,
            start=start,
            end=end_yf,
            multi_level_index=True,       # multi-level khi >1 mã
            progress=False,
            auto_adjust=True,
        )
    except Exception as e:
        print(f"[get_market_context] Lỗi bulk download: {e}")
        return {}
 
    if raw is None or raw.empty:
        return {}
 
    result: dict[str, pd.DataFrame] = {}
    for sym, yf_sym in zip(symbols, yf_symbols):
        try:
            df = raw.xs(yf_sym, axis=1, level=1).copy()
            df = df.reset_index()
            df.columns = [c.lower() for c in df.columns]
            df["date"] = pd.to_datetime(df["date"]).dt.normalize()
            df = df.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)
            if not df.empty:
                result[sym] = df
        except Exception:
            pass  # mã không có dữ liệu → bỏ qua
 
    return result
 
 
def _classify_trend(df: pd.DataFrame) -> str:
    """
    Phân loại xu hướng bằng slope hồi quy tuyến tính trên giá đóng cửa.
    Ngưỡng: 0.1% giá trung bình / phiên.
    Trả về: 'tăng' | 'giảm' | 'đi ngang' | 'không đủ dữ liệu'
    """
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
    threshold = mean_y * 0.001          # 0.1% / phiên
 
    if slope > threshold:
        return "tăng"
    elif slope < -threshold:
        return "giảm"
    else:
        return "đi ngang"
 
 
def _day_change(df: pd.DataFrame, ref_ts: pd.Timestamp) -> dict:
    """Thông tin biến động giá ngày tham chiếu."""
    row = df[df["date"] == ref_ts]
    if row.empty:
        return {
            "date": ref_ts.date(),
            "status": "N/A (không phải ngày giao dịch)",
            "open": None, "close": None, "change_pct": None,
        }
    r = row.iloc[0]
    change_pct = (r["close"] - r["open"]) / r["open"] * 100 if r["open"] else None
    direction = (
        "tăng" if (change_pct or 0) > 0
        else ("giảm" if (change_pct or 0) < 0 else "đi ngang")
    )
    return {
        "date": ref_ts.date(),
        "open": round(float(r["open"]), 2),
        "close": round(float(r["close"]), 2),
        "change_pct": round(change_pct, 2) if change_pct is not None else None,
        "status": direction,
    }
 
def _filter_window(df: pd.DataFrame, ref_dt: datetime, days: int) -> pd.DataFrame:
    """Lọc DataFrame chỉ lấy dữ liệu trong khoảng [ref_dt - days, ref_dt]."""
    start_ts = pd.Timestamp(ref_dt - timedelta(days=days)).normalize()
    end_ts   = pd.Timestamp(ref_dt).normalize()
    return df[(df["date"] >= start_ts) & (df["date"] <= end_ts)].reset_index(drop=True)


def _format_trend_block(df_window: pd.DataFrame, label: str) -> str:
    """Trả về chuỗi mô tả xu hướng cho một cửa sổ thời gian."""
    if df_window.empty:
        return f"  {label}: không đủ dữ liệu"
    trend = _classify_trend(df_window)
    return (
        f"  {label}: **{trend}** "
        f"(đóng cửa đầu kỳ: {df_window.iloc[0]['close']:,.2f} | "
        f"cuối kỳ: {df_window.iloc[-1]['close']:,.2f})"
    )


def get_market_context(
    ticker: Annotated[str, "ticker symbol đang phân tích"],
    curr_date: Annotated[str, "ngày tham chiếu YYYY-mm-dd"],
) -> str:
    """
    Trả về bối cảnh thị trường tại ngày tham chiếu gồm:
      1. Ticker đang phân tích: biến động ngày + xu hướng 7N và 30N
      2. Breadth VN30: bao nhiêu mã tăng / giảm / đi ngang trong 7N và 30N

    Lưu ý: yfinance không cung cấp VN30 Index, nên phần này được bỏ qua.
    """
    SHORT_WINDOW = 7
    LONG_WINDOW  = 30

    ticker = ticker.upper().strip()
    ref_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    # Lấy đủ 30 ngày lịch (≈ ~22 phiên giao dịch) cho cả 2 cửa sổ
    start_dt = ref_dt - timedelta(days=LONG_WINDOW)

    start_str = start_dt.strftime("%Y-%m-%d")
    end_str   = curr_date
    ref_ts    = pd.Timestamp(ref_dt).normalize()

    lines: list[str] = []
    lines.append(f"# Bối cảnh thị trường – ngày tham chiếu {curr_date}")
    lines.append(
        f"# Dữ liệu crawl: {start_str} → {end_str} | "
        f"Phân tích: 7 ngày và 30 ngày\n"
    )
    lines.append(f"## 1. Ticker {ticker}")
    ticker_df = _fetch_ohlcv_single(ticker, start=start_str, end=end_str)

    if ticker_df.empty:
        lines.append(f"  Không lấy được dữ liệu {ticker} từ yfinance.")
    else:
        day_info = _day_change(ticker_df, ref_ts)
        if day_info["change_pct"] is not None:
            lines.append(
                f"  Ngày {day_info['date']}: mở cửa {day_info['open']:,.2f} | "
                f"đóng cửa {day_info['close']:,.2f} | "
                f"thay đổi {day_info['change_pct']:+.2f}% → **{day_info['status']}**"
            )
        else:
            lines.append(f"  Ngày {day_info['date']}: {day_info['status']}")

        ticker_7d  = _filter_window(ticker_df, ref_dt, SHORT_WINDOW)
        lines.append(_format_trend_block(ticker_7d,  "Xu hướng  7 ngày"))
        ticker_30d = _filter_window(ticker_df, ref_dt, LONG_WINDOW)
        lines.append(_format_trend_block(ticker_30d, "Xu hướng 30 ngày"))

    lines.append("")

    lines.append("## 2. Breadth VN30")

    vn30_data = _fetch_ohlcv_bulk(VN30_SYMBOLS, start=start_str, end=end_str)

    def _empty_breadth() -> dict:
        return {"tăng": [], "giảm": [], "đi ngang": [], "lỗi": []}

    breadth_7d  = _empty_breadth()
    breadth_30d = _empty_breadth()

    for sym in VN30_SYMBOLS:
        if sym not in vn30_data:
            breadth_7d["lỗi"].append(sym)
            breadth_30d["lỗi"].append(sym)
            continue

        df_sym = vn30_data[sym]
        for window, breadth in ((SHORT_WINDOW, breadth_7d), (LONG_WINDOW, breadth_30d)):
            df_w = _filter_window(df_sym, ref_dt, window)
            t = _classify_trend(df_w)
            if t in breadth:
                breadth[t].append(sym)
            else:
                breadth["lỗi"].append(sym)

    total = len(VN30_SYMBOLS)

    for window_label, breadth in (("7 ngày", breadth_7d), ("30 ngày", breadth_30d)):
        lines.append(f"\n  ### {window_label} (tổng {total} mã)")
        lines.append(
            f" Tăng   : {len(breadth['tăng']):>3} mã  – {', '.join(breadth['tăng']) or '–'}"
        )
        lines.append(
            f" Giảm   : {len(breadth['giảm']):>3} mã  – {', '.join(breadth['giảm']) or '–'}"
        )
        lines.append(
            f" Đi ngang: {len(breadth['đi ngang']):>3} mã  – {', '.join(breadth['đi ngang']) or '–'}"
        )
        if breadth["lỗi"]:
            lines.append(f" Không lấy được dữ liệu: {', '.join(breadth['lỗi'])}")

    lines.append("")

    # ------------------------------------------------------------------
    # Tóm tắt
    # ------------------------------------------------------------------
    lines.append("## Tóm tắt")
    tick_trend_7d  = _classify_trend(_filter_window(ticker_df, ref_dt, SHORT_WINDOW)) if not ticker_df.empty else "N/A"
    tick_trend_30d = _classify_trend(_filter_window(ticker_df, ref_dt, LONG_WINDOW))  if not ticker_df.empty else "N/A"
    lines.append(
        f"  {ticker} – 7N: {tick_trend_7d} | 30N: {tick_trend_30d}"
    )
    lines.append(
        f"  Breadth VN30  7N: {len(breadth_7d['tăng'])} tăng / {len(breadth_7d['giảm'])} giảm / {len(breadth_7d['đi ngang'])} đi ngang"
    )
    lines.append(
        f"  Breadth VN30 30N: {len(breadth_30d['tăng'])} tăng / {len(breadth_30d['giảm'])} giảm / {len(breadth_30d['đi ngang'])} đi ngang"
    )

    return "\n".join(lines)

def _get_stock_stats_bulk(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicator: Annotated[str, "technical indicator to calculate"],
) -> dict:
    """
    Optimized bulk calculation of stock stats indicators.
    Fetches data once and calculates indicator for all available dates.
    Returns dict mapping date strings to indicator values.
    """
    from .config import get_config
    import pandas as pd
    from stockstats import wrap
    import os
    
    config = get_config()

    # Online data fetching with caching
    today_date = pd.Timestamp.today()
    end_date = today_date
    start_date = today_date - pd.DateOffset(years=15)
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")

    os.makedirs(config["data_cache_dir"], exist_ok=True)

    data_file = os.path.join(
        config["data_cache_dir"],
        f"{symbol}-YFin-data-{start_date_str}-{end_date_str}.csv",
    )

    if os.path.exists(data_file):
        data = pd.read_csv(data_file)
        data["Date"] = pd.to_datetime(data["Date"])
    else:
        downloaded = yf.download(
            _normalize_vn_ticker(symbol),
            start=start_date_str,
            end=end_date_str,
            multi_level_index=False,
            progress=False,
            auto_adjust=True,
        )
        if downloaded is None:
            raise Exception(f"Không có dữ liệu trả về từ yfinance cho mã {symbol}")
        data = downloaded
        data = data.reset_index()
        data.to_csv(data_file, index=False)

    df = wrap(data)
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    
    # Calculate the indicator for all rows at once
    df[indicator]  # This triggers stockstats to calculate the indicator
    
    # Create a dictionary mapping date strings to indicator values
    result_dict = {}
    for _, row in df.iterrows():
        date_str = row["Date"]
        indicator_value = row[indicator]
        
        # Handle NaN/None values
        if pd.isna(indicator_value):
            result_dict[date_str] = "N/A"
        else:
            result_dict[date_str] = str(indicator_value)
    
    return result_dict


def get_stockstats_indicator(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicator: Annotated[str, "technical indicator to get the analysis and report of"],
    curr_date: Annotated[str, "The current trading date you are trading on, YYYY-mm-dd"],
) -> str:

    curr_date_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    curr_date = curr_date_dt.strftime("%Y-%m-%d")

    try:
        indicator_value = StockstatsUtils.get_stock_stats(
            _normalize_vn_ticker(symbol),
            indicator,
            curr_date,
        )
    except Exception as e:
        print(
            f"Lỗi khi lấy dữ liệu chỉ báo {indicator} tại ngày {curr_date}: {e}"
        )
        return ""

    return str(indicator_value)


def get_balance_sheet(
    ticker: Annotated[str, "ticker symbol of the company"],
    freq: Annotated[str, "frequency of data: 'annual' or 'quarterly'"] = "quarterly",
):
    """Get balance sheet data from yfinance."""
    try:
        ticker_obj = yf.Ticker(_normalize_vn_ticker(ticker))
        
        if freq.lower() == "quarterly" or freq.lower() == "quater":
            data = ticker_obj.quarterly_balance_sheet
        else:
            data = ticker_obj.balance_sheet
            
        if data.empty:
            return f"Không có dữ liệu bảng cân đối kế toán cho mã '{ticker}'"

        # Keep only key balance-sheet metrics and translate row labels to Vietnamese.
        key_metrics = [
            # Quy mô và cấu trúc vốn
            "Total Assets",
            "Total Liabilities Net Minority Interest",
            "Stockholders Equity",
            "Net Debt",
            "Total Debt",
            "Long Term Debt",
            "Current Debt",
            # Thanh khoản ngắn hạn
            "Current Assets",
            "Current Liabilities",
            "Working Capital",
            "Cash And Cash Equivalents",
            # Chất lượng tài sản lưu động
            "Accounts Receivable",
            "Inventory",
            # Chất lượng tài sản dài hạn
            "Net PPE",
            "Goodwill",
            "Other Intangible Assets",
            "Investment Properties",
            # Khả năng tích lũy vốn
            "Retained Earnings",
            "Tangible Book Value",
        ]

        metric_labels_vi = {
            "Total Assets": "Tổng tài sản",
            "Total Liabilities Net Minority Interest": "Tổng nợ phải trả (thuần lợi ích thiểu số)",
            "Stockholders Equity": "Vốn chủ sở hữu",
            "Net Debt": "Nợ ròng",
            "Total Debt": "Tổng nợ",
            "Long Term Debt": "Nợ dài hạn",
            "Current Debt": "Nợ vay ngắn hạn",
            "Current Assets": "Tài sản ngắn hạn",
            "Current Liabilities": "Nợ ngắn hạn",
            "Working Capital": "Vốn lưu động",
            "Cash And Cash Equivalents": "Tiền và tương đương tiền",
            "Accounts Receivable": "Phải thu khách hàng",
            "Inventory": "Hàng tồn kho",
            "Net PPE": "Tài sản cố định thuần",
            "Goodwill": "Lợi thế thương mại",
            "Other Intangible Assets": "Tài sản vô hình khác",
            "Investment Properties": "Bất động sản đầu tư",
            "Retained Earnings": "Lợi nhuận giữ lại",
            "Tangible Book Value": "Giá trị sổ sách hữu hình",
        }

        available_metrics = [metric for metric in key_metrics if metric in data.index]
        if available_metrics:
            data = data.loc[available_metrics]
            data = data.rename(index={m: metric_labels_vi.get(m, m) for m in available_metrics})
            
        # Convert to CSV string for consistency with other functions
        csv_string = data.to_csv()
        
        # Add header information
        freq_vn = "năm" if freq == "annual" else "quý"
        header = f"# Dữ liệu bảng cân đối kế toán cho {ticker.upper()} ({freq_vn})\n"
        header += f"# Dữ liệu được lấy vào lúc: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        return header + csv_string
        
    except Exception as e:
        return f"Lỗi khi lấy bảng cân đối kế toán cho {ticker}: {str(e)}"


def get_cashflow(
    ticker: Annotated[str, "ticker symbol of the company"],
    freq: Annotated[str, "frequency of data: 'annual' or 'quarterly'"] = "quarterly",
):
    """Get cash flow data from yfinance."""
    try:
        ticker_obj = yf.Ticker(_normalize_vn_ticker(ticker))
        
        if freq.lower() == "quarterly" or freq.lower() == "quater":
            data = ticker_obj.quarterly_cashflow
        else:
            data = ticker_obj.cashflow
            
        if data.empty:
            return f"Không có dữ liệu lưu chuyển tiền tệ cho mã '{ticker}'"

        # Keep only key cash-flow metrics and translate row labels to Vietnamese.
        key_metrics = [
            # Tổng quan dòng tiền
            "Operating Cash Flow",
            "Investing Cash Flow",
            "Financing Cash Flow",
            "Free Cash Flow",
            # Chất lượng lợi nhuận và dòng tiền hoạt động
            "Net Income From Continuing Operations",
            "Depreciation And Amortization",
            "Change In Working Capital",
            "Change In Receivables",
            "Change In Inventory",
            "Change In Payable",
            # Chi đầu tư
            "Capital Expenditure",
            "Purchase Of PPE",
            "Sale Of PPE",
            # Dòng tiền tài trợ
            "Cash Dividends Paid",
            "Interest Paid Cfo",
            "Taxes Refund Paid",
            "Issuance Of Debt",
            "Repayment Of Debt",
            "Net Issuance Payments Of Debt",
            # Biến động tiền mặt
            "Beginning Cash Position",
            "Changes In Cash",
            "Effect Of Exchange Rate Changes",
            "End Cash Position",
        ]

        metric_labels_vi = {
            "Operating Cash Flow": "Dòng tiền thuần từ hoạt động kinh doanh",
            "Investing Cash Flow": "Dòng tiền thuần từ hoạt động đầu tư",
            "Financing Cash Flow": "Dòng tiền thuần từ hoạt động tài chính",
            "Free Cash Flow": "Dòng tiền tự do",
            "Net Income From Continuing Operations": "Lợi nhuận thuần từ hoạt động liên tục",
            "Depreciation And Amortization": "Khấu hao và phân bổ",
            "Change In Working Capital": "Biến động vốn lưu động",
            "Change In Receivables": "Biến động các khoản phải thu",
            "Change In Inventory": "Biến động hàng tồn kho",
            "Change In Payable": "Biến động các khoản phải trả",
            "Capital Expenditure": "Chi tiêu vốn (CAPEX)",
            "Purchase Of PPE": "Chi mua tài sản cố định",
            "Sale Of PPE": "Thu từ thanh lý tài sản cố định",
            "Cash Dividends Paid": "Cổ tức tiền mặt đã trả",
            "Interest Paid Cfo": "Lãi vay đã trả (CFO)",
            "Taxes Refund Paid": "Thuế đã nộp/hoàn",
            "Issuance Of Debt": "Thu từ phát hành nợ",
            "Repayment Of Debt": "Chi trả nợ gốc",
            "Net Issuance Payments Of Debt": "Thuần phát hành/trả nợ",
            "Beginning Cash Position": "Tiền đầu kỳ",
            "Changes In Cash": "Biến động tiền thuần trong kỳ",
            "Effect Of Exchange Rate Changes": "Ảnh hưởng chênh lệch tỷ giá",
            "End Cash Position": "Tiền cuối kỳ",
        }

        available_metrics = [metric for metric in key_metrics if metric in data.index]
        if available_metrics:
            data = data.loc[available_metrics]
            data = data.rename(index={m: metric_labels_vi.get(m, m) for m in available_metrics})
            
        # Convert to CSV string for consistency with other functions
        csv_string = data.to_csv()
        
        # Add header information
        freq_vn = "năm" if freq == "annual" else "quý"
        header = f"# Dữ liệu lưu chuyển tiền tệ cho {ticker.upper()} ({freq_vn})\n"
        header += f"# Dữ liệu được lấy vào lúc: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        return header + csv_string
        
    except Exception as e:
        return f"Lỗi khi lấy báo cáo lưu chuyển tiền tệ cho {ticker}: {str(e)}"


def get_income_statement(
    ticker: Annotated[str, "ticker symbol of the company"],
    freq: Annotated[str, "frequency of data: 'annual' or 'quarterly'"] = "quarterly",
):
    """Get income statement data from yfinance."""
    try:
        ticker_obj = yf.Ticker(_normalize_vn_ticker(ticker))
        
        if freq.lower() == "quarterly" or freq.lower() == "quater":
            data = ticker_obj.quarterly_income_stmt
        else:
            data = ticker_obj.income_stmt
            
        if data.empty:
            return f"Không có dữ liệu báo cáo kết quả kinh doanh cho mã '{ticker}'"

        # Keep only key income-statement metrics and translate row labels to Vietnamese.
        key_metrics = [
            # Quy mô doanh thu
            "Total Revenue",
            "Operating Revenue",
            # Biên lợi nhuận gộp và chi phí chính
            "Cost Of Revenue",
            "Gross Profit",
            "Operating Expense",
            # Lợi nhuận hoạt động
            "Operating Income",
            "EBIT",
            "EBITDA",
            # Lãi/lỗ tài chính
            "Interest Income",
            "Interest Expense",
            "Net Interest Income",
            # Lợi nhuận trước và sau thuế
            "Pretax Income",
            "Tax Provision",
            "Net Income",
            "Net Income Common Stockholders",
            # Chỉ tiêu trên mỗi cổ phần
            "Basic EPS",
            "Diluted EPS",
            "Basic Average Shares",
            "Diluted Average Shares",
            # Chỉ tiêu chất lượng lợi nhuận
            "Normalized EBITDA",
            "Normalized Income",
            "Total Expenses",
        ]

        metric_labels_vi = {
            "Total Revenue": "Tổng doanh thu",
            "Operating Revenue": "Doanh thu thuần hoạt động",
            "Cost Of Revenue": "Giá vốn hàng bán",
            "Gross Profit": "Lợi nhuận gộp",
            "Operating Expense": "Chi phí hoạt động",
            "Operating Income": "Lợi nhuận hoạt động",
            "EBIT": "Lợi nhuận trước lãi vay và thuế (EBIT)",
            "EBITDA": "Lợi nhuận trước lãi vay, thuế và khấu hao (EBITDA)",
            "Interest Income": "Thu nhập lãi",
            "Interest Expense": "Chi phí lãi vay",
            "Net Interest Income": "Thu nhập lãi thuần",
            "Pretax Income": "Lợi nhuận trước thuế",
            "Tax Provision": "Chi phí thuế TNDN",
            "Net Income": "Lợi nhuận sau thuế",
            "Net Income Common Stockholders": "LNST thuộc cổ đông phổ thông",
            "Basic EPS": "EPS cơ bản",
            "Diluted EPS": "EPS pha loãng",
            "Basic Average Shares": "Số cp bình quân cơ bản",
            "Diluted Average Shares": "Số cp bình quân pha loãng",
            "Normalized EBITDA": "EBITDA chuẩn hóa",
            "Normalized Income": "Lợi nhuận chuẩn hóa",
            "Total Expenses": "Tổng chi phí",
        }

        available_metrics = [metric for metric in key_metrics if metric in data.index]
        if available_metrics:
            data = data.loc[available_metrics]
            data = data.rename(index={m: metric_labels_vi.get(m, m) for m in available_metrics})
            
        # Convert to CSV string for consistency with other functions
        csv_string = data.to_csv()
        
        # Add header information
        freq_vn = "năm" if freq == "annual" else "quý"
        header = f"# Dữ liệu báo cáo kết quả kinh doanh cho {ticker.upper()} ({freq_vn})\n"
        header += f"# Dữ liệu được lấy vào lúc: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        return header + csv_string
        
    except Exception as e:
        return f"Lỗi khi lấy báo cáo kết quả kinh doanh cho {ticker}: {str(e)}"


def get_insider_transactions(
    ticker: Annotated[str, "ticker symbol of the company"]
):
    """Get insider transactions data from yfinance."""
    try:
        ticker_obj = yf.Ticker(_normalize_vn_ticker(ticker))
        data = ticker_obj.insider_transactions
        
        if data is None or data.empty:
            return f"Không có dữ liệu giao dịch nội bộ cho mã '{ticker}'"
            
        # Convert to CSV string for consistency with other functions
        csv_string = data.to_csv()
        
        # Add header information
        header = f"# Dữ liệu giao dịch nội bộ cho {ticker.upper()}\n"
        header += f"# Dữ liệu được lấy vào lúc: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        return header + csv_string
        
    except Exception as e:
        return f"Lỗi khi lấy dữ liệu giao dịch nội bộ cho {ticker}: {str(e)}"


def get_fundamentals(
    ticker: Annotated[str, "ticker symbol of the company"],
):
    """Get consolidated fundamentals report from yfinance."""
    ticker_display = _display_ticker(ticker)
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    try:
        ticker_obj = yf.Ticker(_normalize_vn_ticker(ticker_display))
        info = ticker_obj.info

        if not isinstance(info, dict) or len(info) == 0:
            return f"Không có thông tin tổng quan cổ phiếu cho mã '{ticker_display}'"

        fields_to_show = [
            ("shortName", "Tên ngắn"),
            ("longName", "Tên đầy đủ"),
            ("symbol", "Mã"),
            ("currency", "Đơn vị tiền tệ"),
            ("marketCap", "Vốn hóa thị trường"),
            ("enterpriseValue", "Giá trị doanh nghiệp"),
            ("trailingPE", "P/E hiện tại"),
            ("forwardPE", "P/E dự phóng"),
            ("priceToBook", "P/B"),
            ("beta", "Beta"),
            ("fiftyTwoWeekHigh", "Đỉnh 52 tuần"),
            ("fiftyTwoWeekLow", "Đáy 52 tuần"),
            ("averageVolume", "Khối lượng trung bình"),
            ("sector", "Lĩnh vực"),
            ("industry", "Ngành"),
            ("country", "Quốc gia"),
            ("website", "Website"),
        ]

        lines = []
        for key, label in fields_to_show:
            value = info.get(key)
            if value is not None:
                if key == "symbol" and isinstance(value, str):
                    value = _display_ticker(value)
                lines.append(f"- {label}: {value}")

        if len(lines) == 0:
            return f"Không có trường thông tin tổng quan khả dụng cho mã '{ticker_display}'"

        header = f"# Dữ liệu cơ bản cho {ticker_display}\n"
        header += f"# Dữ liệu được lấy vào lúc: {timestamp}\n\n"
        stock_section = f"# Thông tin tổng quan cổ phiếu cho {ticker_display}\n\n" + "\n".join(lines)

        return header + stock_section
    except Exception as e:
        return f"Lỗi khi lấy dữ liệu cơ bản cho {ticker_display}: {str(e)}"
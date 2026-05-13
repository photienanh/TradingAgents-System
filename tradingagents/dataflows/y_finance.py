from typing import Annotated, Optional
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import yfinance as yf
import pandas as pd
import os
from .config import get_config
from .utils import (
    build_date_window,
    classify_trend,
    day_change,
    filter_window,
    format_trend_block,
    VN30_INDUSTRIES,
    VN30_SYMBOLS
)
import logging

log = logging.getLogger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MARKET_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "market_data")


def _normalize_vn_ticker(symbol: str) -> str:
    ticker = symbol.strip().upper()
    return ticker if ticker.endswith(".VN") else f"{ticker}.VN"


def _display_ticker(symbol: str) -> str:
    ticker = symbol.strip().upper()
    return ticker[:-3] if ticker.endswith(".VN") else ticker


def _read_indicator_from_csv(symbol: str, indicator: str) -> Optional[pd.Series]:
    """
    Đọc cột indicator từ file {symbol}.csv trong data/market_data/.
    Trả về pd.Series với index là datetime, None nếu không tìm thấy.
    """
    path = os.path.join(MARKET_DATA_DIR, f"{symbol.upper()}.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, usecols=lambda c: c in ["time", indicator])
        if indicator not in df.columns:
            return None
        df["time"] = pd.to_datetime(df["time"], errors="coerce").dt.normalize()
        df = df.dropna(subset=["time"]).sort_values("time").set_index("time")
        return df[indicator]
    except Exception:
        return None


def get_YFin_data_online(
    symbol: Annotated[str, "ticker symbol of the company"],
    curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
    look_back_days: Annotated[int, "Number of days to look back"],
):
    try:
        _, _, start_date, end_date = build_date_window(curr_date, look_back_days)
    except ValueError:
        return "look_back_days phải >= 0"

    ticker = yf.Ticker(_normalize_vn_ticker(symbol))
    end_date_yf = (
        datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
    ).strftime("%Y-%m-%d")

    data = ticker.history(start=start_date, end=end_date_yf)

    if data.empty:
        return f"Không có dữ liệu cho mã '{symbol}' từ {start_date} đến {end_date}"

    if isinstance(data.index, pd.DatetimeIndex) and data.index.tz is not None:
        data.index = data.index.tz_localize(None)

    data.drop(columns=["Dividends", "Stock Splits"], inplace=True, errors="ignore")
    csv_string = data.to_csv()

    header = f"# Dữ liệu cổ phiếu cho {symbol.upper()} từ {start_date} đến {end_date}\n"
    header += f"# Tổng số bản ghi: {len(data)}\n"
    header += f"# Dữ liệu được lấy vào lúc: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    return header + csv_string


def get_indicators(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicator: Annotated[str, "technical indicator to get the analysis and report of"],
    curr_date: Annotated[str, "The current trading date you are trading on, YYYY-mm-dd"],
    look_back_days: Annotated[int, "how many days to look back"],
) -> str:
    best_ind_params = {
        "sma_50": (
            "50 SMA: Chỉ báo xu hướng trung hạn. "
            "Cách dùng: Xác định hướng xu hướng và làm vùng hỗ trợ/kháng cự động. "
            "Lưu ý: Có độ trễ so với giá, nên kết hợp thêm chỉ báo nhanh để vào lệnh kịp thời."
        ),
        "sma_200": (
            "200 SMA: Mốc tham chiếu xu hướng dài hạn. "
            "Cách dùng: Xác nhận xu hướng tổng thể và nhận diện mô hình golden/death cross. "
            "Lưu ý: Phản ứng chậm, phù hợp xác nhận chiến lược hơn là tín hiệu vào lệnh thường xuyên."
        ),
        "ema_10": (
            "10 EMA: Đường trung bình ngắn hạn phản ứng nhanh. "
            "Cách dùng: Bắt các thay đổi động lượng sớm và tìm điểm vào lệnh tiềm năng. "
            "Lưu ý: Dễ nhiễu khi thị trường đi ngang, nên kết hợp với MA dài hơn để lọc tín hiệu giả."
        ),
        "macd": (
            "MACD: Đo động lượng thông qua chênh lệch giữa các EMA. "
            "Cách dùng: Theo dõi giao cắt và phân kỳ để nhận diện khả năng đổi xu hướng. "
            "Lưu ý: Nên xác nhận thêm bằng chỉ báo khác khi thị trường biến động thấp hoặc đi ngang."
        ),
        "macd_signal": (
            "MACD Signal: Đường EMA làm mượt của MACD. "
            "Cách dùng: Dùng giao cắt giữa MACD và Signal để kích hoạt tín hiệu giao dịch. "
            "Lưu ý: Nên nằm trong chiến lược tổng thể để giảm tín hiệu sai."
        ),
        "macd_hist": (
            "MACD Histogram: Thể hiện độ lệch giữa MACD và đường Signal. "
            "Cách dùng: Quan sát độ mạnh động lượng và phát hiện phân kỳ sớm. "
            "Lưu ý: Có thể biến động mạnh, nên bổ sung bộ lọc khi thị trường chạy nhanh."
        ),
        "rsi_14": (
            "RSI: Đo động lượng để nhận biết vùng quá mua/quá bán. "
            "Cách dùng: Dùng ngưỡng 70/30 và theo dõi phân kỳ để tìm tín hiệu đảo chiều. "
            "Lưu ý: Trong xu hướng mạnh RSI có thể neo ở vùng cực trị, cần đối chiếu thêm phân tích xu hướng."
        ),
        "bb_middle": (
            "Bollinger Middle: SMA 20 làm đường cơ sở của Bollinger Bands. "
            "Cách dùng: Là mốc động để đánh giá dao động giá. "
            "Lưu ý: Kết hợp với dải trên/dưới để nhận diện breakout hoặc đảo chiều hiệu quả hơn."
        ),
        "bb_upper": (
            "Bollinger Upper Band: Thường nằm trên đường giữa khoảng 2 độ lệch chuẩn. "
            "Cách dùng: Gợi ý vùng quá mua và vùng có thể bứt phá. "
            "Lưu ý: Cần xác nhận bằng công cụ khác; trong xu hướng mạnh giá có thể bám dải trên lâu."
        ),
        "bb_lower": (
            "Bollinger Lower Band: Thường nằm dưới đường giữa khoảng 2 độ lệch chuẩn. "
            "Cách dùng: Gợi ý vùng quá bán tiềm năng. "
            "Lưu ý: Nên có phân tích bổ sung để tránh bắt nhầm tín hiệu đảo chiều giả."
        ),
        "atr_14": (
            "ATR: Đo biến động bằng trung bình True Range 14 phiên. "
            "Cách dùng: Đặt stop-loss và điều chỉnh khối lượng vị thế theo mức biến động hiện tại. "
            "Lưu ý: Đây là chỉ báo phản ứng, nên dùng trong chiến lược quản trị rủi ro tổng thể."
        ),
        "vwma_20": (
            "VWMA: Đường trung bình động có trọng số theo khối lượng 20 phiên. "
            "Cách dùng: Xác nhận xu hướng bằng cách kết hợp biến động giá với dữ liệu khối lượng. "
            "Lưu ý: Cẩn trọng khi có đột biến volume vì có thể làm lệch kết quả."
        ),
        "mfi_14": (
            "MFI: Money Flow Index — chỉ báo động lượng dùng cả giá và khối lượng. "
            "Cách dùng: Xác định quá mua (>80) hoặc quá bán (<20), xác nhận độ mạnh xu hướng. "
            "Lưu ý: Nên dùng cùng RSI hoặc MACD để xác nhận; phân kỳ giữa giá và MFI báo hiệu đảo chiều."
        ),
    }

    if indicator not in best_ind_params:
        raise ValueError(
            f"Chỉ báo {indicator} chưa được hỗ trợ. Vui lòng chọn một trong: {list(best_ind_params.keys())}"
        )

    end_date = curr_date
    curr_date_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    before = curr_date_dt - relativedelta(days=look_back_days)

    # Đọc từ file CSV trước
    series = _read_indicator_from_csv(symbol, indicator)

    if series is not None:
        mask = (series.index >= pd.Timestamp(before)) & (series.index <= pd.Timestamp(curr_date_dt))
        series_window = series[mask].sort_index(ascending=False)

        ind_string = ""
        existing_dates = set()
        for date_idx, value in series_window.items():
            date_str = date_idx.strftime("%Y-%m-%d")
            existing_dates.add(date_str)
            if pd.isna(value):
                ind_string += f"{date_str}: N/A\n"
            else:
                ind_string += f"{date_str}: {value}\n"

        # Điền ngày không có giao dịch
        current_dt = curr_date_dt
        extra_lines = []
        while current_dt >= before:
            date_str = current_dt.strftime("%Y-%m-%d")
            if date_str not in existing_dates:
                extra_lines.append(f"{date_str}: N/A: Không phải ngày giao dịch\n")
            current_dt -= timedelta(days=1)

        # Merge và sort
        all_lines = (ind_string + "".join(extra_lines)).strip().split("\n")
        all_lines = [l for l in all_lines if l]
        all_lines.sort(key=lambda x: x.split(":")[0], reverse=True)
        ind_string = "\n".join(all_lines) + "\n"

    else:
        ind_string = f"N/A: Không tìm thấy dữ liệu {indicator} cho {symbol}.\n"

    result_str = (
        f"## Giá trị {indicator} từ {before.strftime('%Y-%m-%d')} đến {end_date}:\n\n"
        + ind_string
        + "\n\n"
        + best_ind_params.get(indicator, "Không có mô tả cho chỉ báo này.")
    )

    return result_str


def _fetch_ohlcv_single(symbol: str, start: str, end: str) -> pd.DataFrame:
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
        df.columns = [c.lower() for c in df.columns]
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df = df.sort_values("date").reset_index(drop=True)
        return df
    except Exception as e:
        print(f"[get_market_context] Lỗi fetch {symbol}: {e}")
        return pd.DataFrame()


def _fetch_ohlcv_bulk(symbols: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    end_yf = (datetime.strptime(end, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    yf_symbols = [_normalize_vn_ticker(s) for s in symbols]

    try:
        raw = yf.download(
            yf_symbols,
            start=start,
            end=end_yf,
            multi_level_index=True,
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
            pass

    return result


def get_market_context(
    ticker: Annotated[str, "ticker symbol đang phân tích"],
    curr_date: Annotated[str, "ngày tham chiếu YYYY-mm-dd"],
) -> str:
    SHORT_WINDOW = 7
    LONG_WINDOW  = 30

    ticker = ticker.upper().strip()
    ref_dt = datetime.strptime(curr_date, "%Y-%m-%d")
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
        day_info = day_change(ticker_df, ref_ts)
        if day_info["change_pct"] is not None:
            lines.append(
                f"  Ngày {day_info['date']}: mở cửa {day_info['open']:,.2f} | "
                f"đóng cửa {day_info['close']:,.2f} | "
                f"thay đổi {day_info['change_pct']:+.2f}% → **{day_info['status']}**"
            )
        else:
            lines.append(f"  Ngày {day_info['date']}: {day_info['status']}")

        ticker_7d  = filter_window(ticker_df, ref_dt, SHORT_WINDOW)
        lines.append(format_trend_block(ticker_7d,  "Xu hướng  7 ngày"))
        ticker_30d = filter_window(ticker_df, ref_dt, LONG_WINDOW)
        lines.append(format_trend_block(ticker_30d, "Xu hướng 30 ngày"))

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
            df_w = filter_window(df_sym, ref_dt, window)
            t = classify_trend(df_w)
            if t in breadth:
                breadth[t].append(sym)
            else:
                breadth["lỗi"].append(sym)

    total = len(VN30_SYMBOLS)

    for window_label, breadth in (("7 ngày", breadth_7d), ("30 ngày", breadth_30d)):
        lines.append(f"\n  ### {window_label} (tổng {total} mã)")
        lines.append(f" Tăng   : {len(breadth['tăng']):>3} mã  – {', '.join(breadth['tăng']) or '–'}")
        lines.append(f" Giảm   : {len(breadth['giảm']):>3} mã  – {', '.join(breadth['giảm']) or '–'}")
        lines.append(f" Đi ngang: {len(breadth['đi ngang']):>3} mã  – {', '.join(breadth['đi ngang']) or '–'}")
        if breadth["lỗi"]:
            lines.append(f" Không lấy được dữ liệu: {', '.join(breadth['lỗi'])}")

    lines.append("")

    industry_name = None
    industry_symbols: list[str] = []
    for name, symbols in VN30_INDUSTRIES.items():
        if ticker in symbols:
            industry_name = name
            industry_symbols = symbols
            break

    industry_eval = None
    if industry_name and len(industry_symbols) >= 2:
        lines.append("## 3. Đánh giá cùng nhóm ngành (VN30)")
        peers = [s for s in industry_symbols if s != ticker]

        def _industry_breadth(window_days: int) -> dict:
            b = {"tăng": [], "giảm": [], "đi ngang": [], "lỗi": []}
            for sym in peers:
                if sym not in vn30_data:
                    b["lỗi"].append(sym)
                    continue
                df_w = filter_window(vn30_data[sym], ref_dt, window_days)
                t = classify_trend(df_w)
                if t in b:
                    b[t].append(sym)
                else:
                    b["lỗi"].append(sym)
            return b

        industry_7d = _industry_breadth(SHORT_WINDOW)
        industry_30d = _industry_breadth(LONG_WINDOW)
        industry_eval = {
            "name": industry_name,
            "peer_count": len(peers),
            "7d": industry_7d,
            "30d": industry_30d,
        }

        lines.append(f"  Ngành: {industry_name}")
        for label, b in (("7 ngày", industry_7d), ("30 ngày", industry_30d)):
            lines.append(f"  {label}: {len(b['tăng'])} tăng / {len(b['giảm'])} giảm / {len(b['đi ngang'])} đi ngang")
            if b["tăng"]:
                lines.append(f"    Mã tăng: {', '.join(b['tăng'])}")
            if b["giảm"]:
                lines.append(f"    Mã giảm: {', '.join(b['giảm'])}")
            if b["đi ngang"]:
                lines.append(f"    Mã đi ngang: {', '.join(b['đi ngang'])}")
            if b["lỗi"]:
                lines.append(f"    Không lấy được dữ liệu: {', '.join(b['lỗi'])}")
        lines.append("")

    lines.append("## Tóm tắt")
    tick_trend_7d  = classify_trend(filter_window(ticker_df, ref_dt, SHORT_WINDOW)) if not ticker_df.empty else "N/A"
    tick_trend_30d = classify_trend(filter_window(ticker_df, ref_dt, LONG_WINDOW))  if not ticker_df.empty else "N/A"
    lines.append(f"  {ticker} – 7N: {tick_trend_7d} | 30N: {tick_trend_30d}")
    lines.append(
        f"  Breadth VN30  7N: {len(breadth_7d['tăng'])} tăng / {len(breadth_7d['giảm'])} giảm / {len(breadth_7d['đi ngang'])} đi ngang"
    )
    lines.append(
        f"  Breadth VN30 30N: {len(breadth_30d['tăng'])} tăng / {len(breadth_30d['giảm'])} giảm / {len(breadth_30d['đi ngang'])} đi ngang"
    )
    if industry_eval:
        lines.append(
            f"  Nhóm {industry_eval['name']} (không gồm {ticker}) 7N: "
            f"{len(industry_eval['7d']['tăng'])} tăng / {len(industry_eval['7d']['giảm'])} giảm / {len(industry_eval['7d']['đi ngang'])} đi ngang"
        )
        lines.append(
            f"  Nhóm {industry_eval['name']} (không gồm {ticker}) 30N: "
            f"{len(industry_eval['30d']['tăng'])} tăng / {len(industry_eval['30d']['giảm'])} giảm / {len(industry_eval['30d']['đi ngang'])} đi ngang"
        )

    return "\n".join(lines)


def get_balance_sheet(
    ticker: Annotated[str, "ticker symbol of the company"],
    freq: Annotated[str, "frequency of data: 'annual' or 'quarterly'"] = "quarterly",
):
    try:
        ticker_obj = yf.Ticker(_normalize_vn_ticker(ticker))

        if freq.lower() in ("quarterly", "quater"):
            data = ticker_obj.quarterly_balance_sheet
        else:
            data = ticker_obj.balance_sheet

        if data.empty:
            return f"Không có dữ liệu bảng cân đối kế toán cho mã '{ticker}'"

        key_metrics = [
            "Total Assets", "Total Liabilities Net Minority Interest", "Stockholders Equity",
            "Net Debt", "Total Debt", "Long Term Debt", "Current Debt",
            "Current Assets", "Current Liabilities", "Working Capital", "Cash And Cash Equivalents",
            "Accounts Receivable", "Inventory", "Net PPE", "Goodwill",
            "Other Intangible Assets", "Investment Properties", "Retained Earnings", "Tangible Book Value",
        ]
        metric_labels_vi = {
            "Total Assets": "Tổng tài sản",
            "Total Liabilities Net Minority Interest": "Tổng nợ phải trả",
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
        available_metrics = [m for m in key_metrics if m in data.index]
        if available_metrics:
            data = data.loc[available_metrics]
            data = data.rename(index={m: metric_labels_vi.get(m, m) for m in available_metrics})

        csv_string = data.to_csv()
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
    try:
        ticker_obj = yf.Ticker(_normalize_vn_ticker(ticker))

        if freq.lower() in ("quarterly", "quater"):
            data = ticker_obj.quarterly_cashflow
        else:
            data = ticker_obj.cashflow

        if data.empty:
            return f"Không có dữ liệu lưu chuyển tiền tệ cho mã '{ticker}'"

        key_metrics = [
            "Operating Cash Flow", "Investing Cash Flow", "Financing Cash Flow", "Free Cash Flow",
            "Net Income From Continuing Operations", "Depreciation And Amortization",
            "Change In Working Capital", "Capital Expenditure", "Cash Dividends Paid",
            "Beginning Cash Position", "Changes In Cash", "End Cash Position",
        ]
        metric_labels_vi = {
            "Operating Cash Flow": "Dòng tiền thuần từ hoạt động kinh doanh",
            "Investing Cash Flow": "Dòng tiền thuần từ hoạt động đầu tư",
            "Financing Cash Flow": "Dòng tiền thuần từ hoạt động tài chính",
            "Free Cash Flow": "Dòng tiền tự do",
            "Net Income From Continuing Operations": "Lợi nhuận thuần từ hoạt động liên tục",
            "Depreciation And Amortization": "Khấu hao và phân bổ",
            "Change In Working Capital": "Biến động vốn lưu động",
            "Capital Expenditure": "Chi tiêu vốn (CAPEX)",
            "Cash Dividends Paid": "Cổ tức tiền mặt đã trả",
            "Beginning Cash Position": "Tiền đầu kỳ",
            "Changes In Cash": "Biến động tiền thuần trong kỳ",
            "End Cash Position": "Tiền cuối kỳ",
        }
        available_metrics = [m for m in key_metrics if m in data.index]
        if available_metrics:
            data = data.loc[available_metrics]
            data = data.rename(index={m: metric_labels_vi.get(m, m) for m in available_metrics})

        csv_string = data.to_csv()
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
    try:
        ticker_obj = yf.Ticker(_normalize_vn_ticker(ticker))

        if freq.lower() in ("quarterly", "quater"):
            data = ticker_obj.quarterly_income_stmt
        else:
            data = ticker_obj.income_stmt

        if data.empty:
            return f"Không có dữ liệu báo cáo kết quả kinh doanh cho mã '{ticker}'"

        key_metrics = [
            "Total Revenue", "Operating Revenue", "Cost Of Revenue", "Gross Profit",
            "Operating Expense", "Operating Income", "EBIT", "EBITDA",
            "Interest Income", "Interest Expense", "Pretax Income",
            "Tax Provision", "Net Income", "Net Income Common Stockholders",
            "Basic EPS", "Diluted EPS",
        ]
        metric_labels_vi = {
            "Total Revenue": "Tổng doanh thu",
            "Operating Revenue": "Doanh thu thuần hoạt động",
            "Cost Of Revenue": "Giá vốn hàng bán",
            "Gross Profit": "Lợi nhuận gộp",
            "Operating Expense": "Chi phí hoạt động",
            "Operating Income": "Lợi nhuận hoạt động",
            "EBIT": "EBIT",
            "EBITDA": "EBITDA",
            "Interest Income": "Thu nhập lãi",
            "Interest Expense": "Chi phí lãi vay",
            "Pretax Income": "Lợi nhuận trước thuế",
            "Tax Provision": "Chi phí thuế TNDN",
            "Net Income": "Lợi nhuận sau thuế",
            "Net Income Common Stockholders": "LNST thuộc cổ đông phổ thông",
            "Basic EPS": "EPS cơ bản",
            "Diluted EPS": "EPS pha loãng",
        }
        available_metrics = [m for m in key_metrics if m in data.index]
        if available_metrics:
            data = data.loc[available_metrics]
            data = data.rename(index={m: metric_labels_vi.get(m, m) for m in available_metrics})

        csv_string = data.to_csv()
        freq_vn = "năm" if freq == "annual" else "quý"
        header = f"# Dữ liệu báo cáo kết quả kinh doanh cho {ticker.upper()} ({freq_vn})\n"
        header += f"# Dữ liệu được lấy vào lúc: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        return header + csv_string

    except Exception as e:
        return f"Lỗi khi lấy báo cáo kết quả kinh doanh cho {ticker}: {str(e)}"


def get_insider_transactions(
    ticker: Annotated[str, "ticker symbol of the company"]
):
    try:
        ticker_obj = yf.Ticker(_normalize_vn_ticker(ticker))
        data = ticker_obj.insider_transactions

        if data is None or data.empty:
            return f"Không có dữ liệu giao dịch nội bộ cho mã '{ticker}'"

        csv_string = data.to_csv()
        header = f"# Dữ liệu giao dịch nội bộ cho {ticker.upper()}\n"
        header += f"# Dữ liệu được lấy vào lúc: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        return header + csv_string

    except Exception as e:
        return f"Lỗi khi lấy dữ liệu giao dịch nội bộ cho {ticker}: {str(e)}"


def get_fundamentals(
    ticker: Annotated[str, "ticker symbol of the company"],
):
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
from typing import Annotated, Optional
from datetime import datetime
from dateutil.relativedelta import relativedelta
from vnstock import Quote, Finance, Company, Listing
import pandas as pd
import os
import re
from stockstats import wrap
from .config import get_config

class Vnstock_Stats:
    @staticmethod
    def get_stock_stats(
        symbol: Annotated[str, "ticker symbol for the company"],
        indicator: Annotated[
            str, "quantitative indicators based off of the stock data for the company"
        ],
        curr_date: Annotated[
            str, "curr date for retrieving stock price data, YYYY-mm-dd"
        ],
    ):
        config = get_config()
        df = None
        data = None
        
        today_date = pd.Timestamp.today()
        curr_date_dt = pd.to_datetime(curr_date)
        end_date = today_date
        start_date = today_date - pd.DateOffset(years=15)
        start_date = start_date.strftime("%Y-%m-%d")
        end_date = end_date.strftime("%Y-%m-%d")

        os.makedirs(config["data_cache_dir"], exist_ok=True)
        data_file = os.path.join(
            config["data_cache_dir"],
            f"{symbol}-data-{start_date}-{end_date}.csv",
        )

        if os.path.exists(data_file):
            data = pd.read_csv(data_file)
            data["Date"] = pd.to_datetime(data["Date"])
        else:
            data = Quote(symbol=symbol).history(start=start_date, end=end_date)
            data.rename(columns={"time": "Date", "open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}, inplace=True)
            data = data[["Date", "Close", "High", "Low", "Open", "Volume"]]
            data["Date"] = data["Date"].dt.strftime("%Y-%m-%d")
            data.to_csv(data_file, index=False)

        df = wrap(data)
        curr_date_key = curr_date_dt.strftime("%Y-%m-%d")
        df[indicator]
        matching_rows = df[df["Date"].str.startswith(curr_date_key)]
        if not matching_rows.empty:
            indicator_value = matching_rows[indicator].values[0]
            return indicator_value
        else:
            return "N/A: Không phải ngày giao dịch (cuối tuần hoặc ngày lễ)"
        
def get_stock_data(
    symbol: Annotated[str, "ticker symbol"],
    start_date: Annotated[str, "start date for historical data, yyyy-mm-dd"],
    end_date: Annotated[str, "end date for historical data, yyyy-mm-dd"],
):
    symbol = symbol.upper()
    try:
        data = Quote(symbol=symbol).history(start_date, end_date)
        if data.empty:
            return f"Không có lịch sử dữ liệu cho {symbol} từ {start_date} đến {end_date}."
        if isinstance(data.index, pd.DatetimeIndex) and data.index.tz is not None:
            data.index = data.index.tz_localize(None)
        
        csv_string = data.to_csv(index=False)

        header = f"# Dữ liệu cổ phiếu cho {symbol.upper()} từ {start_date} đến {end_date}\n"
        header += f"# Tổng số bản ghi: {len(data)}\n"
        header += f"# Dữ liệu được lấy vào lúc: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        return header + csv_string
    except ValueError:
        return f"Không tìm thấy mã cổ phiếu {symbol}."
    
def get_indicators(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicator: Annotated[str, "technical indicator to get the analysis and report of"],
    curr_date: Annotated[str, "The current trading date you are trading on, YYYY-mm-dd"],
    look_back_days: Annotated[int, "how many days to look back"],
) -> str:
    best_ind_params = {
        "close_50_sma": "50 SMA: Đường trung bình động 50 phiên, phản ánh xu hướng trung hạn.",
        "close_200_sma": "200 SMA: Đường trung bình động 200 phiên, phản ánh xu hướng dài hạn.",
        "close_10_ema": "10 EMA: Đường EMA 10 phiên, nhạy hơn với biến động giá ngắn hạn.",
        "macd": "MACD: Chỉ báo động lượng dựa trên chênh lệch giữa các đường EMA.",
        "macds": "MACD Signal: Đường tín hiệu của MACD, dùng để xác nhận điểm giao cắt.",
        "macdh": "MACD Histogram: Biểu đồ chênh lệch giữa MACD và đường tín hiệu.",
        "rsi": "RSI: Chỉ báo sức mạnh tương đối, đánh giá trạng thái quá mua/quá bán.",
        "boll": "Bollinger Middle: Đường giữa của Bollinger Bands (thường là SMA 20).",
        "boll_ub": "Bollinger Upper Band: Dải trên Bollinger, thường cách đường giữa 2 độ lệch chuẩn.",
        "boll_lb": "Bollinger Lower Band: Dải dưới Bollinger, thường cách đường giữa 2 độ lệch chuẩn.",
        "atr": "ATR: Chỉ báo đo mức độ biến động giá trung bình.",
        "vwma": "VWMA: Đường trung bình động có trọng số theo khối lượng giao dịch.",
        "mfi": "MFI: Chỉ báo dòng tiền dựa trên cả giá và khối lượng.",
    }

    if indicator not in best_ind_params:
        raise ValueError(
            f"Chỉ báo {indicator} chưa được hỗ trợ. Vui lòng chọn một trong các chỉ báo: {list(best_ind_params.keys())}"
        )

    symbol = symbol.upper()
    end_date = curr_date
    curr_date_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    before = curr_date_dt - relativedelta(days=look_back_days)

    try:
        indicator_data = _get_stock_stats_bulk(symbol, indicator, curr_date)

        current_dt = curr_date_dt
        date_values = []
        while current_dt >= before:
            date_str = current_dt.strftime("%Y-%m-%d")
            if date_str in indicator_data:
                indicator_value = indicator_data[date_str]
            else:
                indicator_value = "N/A: Không phải ngày giao dịch (cuối tuần hoặc ngày lễ)"
            date_values.append((date_str, indicator_value))
            current_dt = current_dt - relativedelta(days=1)

        ind_string = ""
        for date_str, value in date_values:
            ind_string += f"{date_str}: {value}\n"

    except Exception as e:
        print(f"Lỗi khi lấy dữ liệu chỉ báo theo dạng bulk từ vnstock: {e}")
        ind_string = ""
        current_dt = curr_date_dt
        while current_dt >= before:
            indicator_value = get_stockstats_indicator(
                symbol,
                indicator,
                current_dt.strftime("%Y-%m-%d"),
            )
            ind_string += f"{current_dt.strftime('%Y-%m-%d')}: {indicator_value}\n"
            current_dt = current_dt - relativedelta(days=1)

    result_str = (
        f"## Giá trị {indicator} từ {before.strftime('%Y-%m-%d')} đến {end_date}:\n\n"
        + ind_string
        + "\n\n"
        + best_ind_params.get(indicator, "Không có mô tả cho chỉ báo này.")
    )

    return result_str


def _get_stock_stats_bulk(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicator: Annotated[str, "technical indicator to calculate"],
    curr_date: Annotated[str, "current date for reference (not used)"],
) -> dict:
    """Bulk calculation for vnstock indicators with local cache."""
    from stockstats import wrap
    from .config import get_config

    config = get_config()

    today_date = pd.Timestamp.today()
    start_date = (today_date - pd.DateOffset(years=15)).strftime("%Y-%m-%d")
    end_date = today_date.strftime("%Y-%m-%d")

    os.makedirs(config["data_cache_dir"], exist_ok=True)
    data_file = os.path.join(
        config["data_cache_dir"],
        f"{symbol}-data-{start_date}-{end_date}.csv",
    )

    if os.path.exists(data_file):
        data = pd.read_csv(data_file)
    else:
        data = Quote(symbol=symbol).history(start=start_date, end=end_date)
        if data is None or data.empty:
            raise ValueError(f"Không có dữ liệu vnstock trả về cho mã {symbol}")

        data.rename(
            columns={
                "time": "Date",
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            },
            inplace=True,
        )

        required_cols = ["Date", "Close", "High", "Low", "Open", "Volume"]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Thiếu các cột OHLCV bắt buộc từ vnstock: {missing_cols}")

        data = data[required_cols]
        data.to_csv(data_file, index=False)

    if "Date" not in data.columns:
        raise ValueError("Không tìm thấy cột Date trong dữ liệu vnstock")

    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data = data.dropna(subset=["Date"]).copy()
    data["Date"] = data["Date"].dt.strftime("%Y-%m-%d")

    df = wrap(data)
    df[indicator]

    result_dict = {}
    for _, row in df.iterrows():
        date_str = row["Date"]
        indicator_value = row[indicator]
        result_dict[date_str] = "N/A" if pd.isna(indicator_value) else str(indicator_value)

    return result_dict

def get_stockstats_indicator(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicator: Annotated[str, "technical indicator to get the analysis and report of"],
    curr_date: Annotated[str, "The current trading date you are trading on, YYYY-mm-dd"],
) -> str:

    curr_date_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    curr_date = curr_date_dt.strftime("%Y-%m-%d")

    try:
        indicator_value = Vnstock_Stats.get_stock_stats(
            symbol,
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
    symbol: Annotated[str, "ticker symbol of the company"],
    freq: Annotated[str, "frequency of data: 'year' or 'quarter'"] = "quarter",
    curr_date: Annotated[Optional[str], "current date (not used)"] = None
):
    try:
        if freq.lower() == "quaterly":
            freq = "quarter"
        elif freq.lower() == "annual":
            freq = "year"

        try:
            finance = Finance(symbol=symbol.upper(), source="KBS")
            data = finance.balance_sheet(period=freq)
        except Exception as e:
            print(f"Lỗi khi lấy dữ liệu bảng cân đối kế toán từ KBS cho {symbol}: {e}, thử lại với VCI")
            finance = Finance(symbol=symbol.upper(), source="VCI")
            data = finance.balance_sheet(period=freq)

        if data.empty:
            return f"Không có dữ liệu bảng cân đối kế toán cho mã '{symbol}'"
        
        data.dropna(inplace=True)
        data = data.drop(columns=["item_id"])
        # Convert to CSV string for consistency with other functions
        csv_string = data.to_csv(index=False)
        
        # Add header information
        freq_vn = "năm" if freq == "year" else "quý"
        header = f"# Dữ liệu bảng cân đối kế toán cho {symbol.upper()} theo ({freq_vn})\n"
        header += f"# Dữ liệu được lấy vào lúc: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        return header + csv_string
        
    except Exception as e:
        return f"Lỗi khi lấy bảng cân đối kế toán cho {symbol}: {str(e)}"
    
def get_cashflow(
    symbol: Annotated[str, "ticker symbol of the company"],
    freq: Annotated[str, "frequency of data: 'year' or 'quarter'"] = "year",
    curr_date: Annotated[Optional[str], "current date (not used for yfinance)"] = None
):
    try:
        if freq.lower() == "quaterly":
            freq = "quarter"
        elif freq.lower() == "annual":
            freq = "year"

        try:
            finance = Finance(symbol=symbol.upper(), source="KBS")
            data = finance.cash_flow(period=freq)
        except Exception as e:
            print(f"Lỗi khi lấy dữ liệu lưu chuyển tiền tệ từ KBS cho {symbol}: {e}, thử lại với VCI")
            finance = Finance(symbol=symbol.upper(), source="VCI")
            data = finance.cash_flow(period=freq)

        if data.empty:
            return f"Không có dữ liệu báo cáo lưu chuyển tiền tệ cho mã '{symbol}'"
        
        if freq == "year":
            data.dropna(inplace=True)
        
        if "item_id" in data.columns:
            data = data.drop(columns=["item_id"])
        csv_string = data.to_csv(index=False)

        # Add header information
        freq_vn = "năm" if freq == "year" else "quý"
        header = f"# Dữ liệu lưu chuyển tiền tệ cho {symbol.upper()} theo ({freq_vn})\n"
        header += f"# Dữ liệu được lấy vào lúc: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        return header + csv_string
        
    except Exception as e:
        return f"Lỗi khi lấy báo cáo lưu chuyển tiền tệ cho {symbol}: {str(e)}"

def get_income_statement(
    symbol: Annotated[str, "ticker symbol of the company"],
    freq: Annotated[str, "frequency of data: 'year' or 'quarter'"] = "quarter",
    curr_date: Annotated[Optional[str], "current date (not used)"] = None
):

    try:
        if freq.lower() == "quaterly":
            freq = "quarter"
        elif freq.lower() == "annual":
            freq = "year"
        
        try:
            finance = Finance(symbol=symbol.upper(), source="KBS")
            data = finance.income_statement(period=freq)
        except Exception as e:
            print(f"Lỗi khi lấy dữ liệu báo cáo kết quả kinh doanh từ KBS cho {symbol}: {e}, thử lại với VCI")
            finance = Finance(symbol=symbol.upper(), source="VCI")
            data = finance.income_statement(period=freq)

        if data.empty:
            return f"Không có dữ liệu bảng cân đối kế toán cho mã '{symbol}'"
        
        data.dropna(inplace=True)
        data = data.drop(columns=["item_id"])

        # Convert to CSV string for consistency with other functions
        csv_string = data.to_csv(index=False)
        
        # Add header information
        freq_vn = "năm" if freq == "year" else "quý"
        header = f"# Dữ liệu báo cáo kết quả kinh doanh cho {symbol.upper()} theo ({freq_vn})\n"
        header += f"# Dữ liệu được lấy vào lúc: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        return header + csv_string
        
    except Exception as e:
        return f"Lỗi khi lấy báo cáo kết quả kinh doanh cho {symbol}: {str(e)}"

def get_fundamentals(
    symbol: Annotated[str, "ticker symbol of the company"],
    curr_date: Annotated[Optional[str], "current date (not used)"] = None,
):
    try:
        ticker = symbol.upper()

        listing_name = None
        try:
            listing_df = Listing(source="KBS").all_symbols()
            if listing_df is not None and not listing_df.empty and "symbol" in listing_df.columns:
                matched = listing_df[
                    listing_df["symbol"].astype(str).str.upper() == ticker
                ]
                if not matched.empty:
                    if "organ_name" in matched.columns and pd.notna(matched.iloc[0]["organ_name"]):
                        listing_name = str(matched.iloc[0]["organ_name"]).strip()
        except Exception:
            listing_name = None

        company = Company(symbol=ticker, source="KBS")
        overview_df = company.overview()
        overview = {}
        if overview_df is not None and not overview_df.empty:
            overview = overview_df.iloc[0].to_dict()

        if listing_name:
            overview["company_name"] = listing_name
        ratio_source = "KBS"
        try:
            ratio_df = Finance(symbol=ticker, source="KBS").ratio(period="quarter")
        except Exception:
            ratio_source = "VCI"
            ratio_df = Finance(symbol=ticker, source="VCI").ratio(period="quarter")
        ratio_map = {}
        latest_quarter = None
        if ratio_df is not None and not ratio_df.empty:
            # KBS trả về dạng bảng chỉ tiêu theo hàng (item_id + các cột kỳ báo cáo),
            # còn VCI trả về MultiIndex theo nhóm chỉ tiêu. Parse theo source để giữ fallback ổn định.
            if ratio_source == "KBS":
                quarter_cols = [
                    c for c in ratio_df.columns
                    if c not in ["item", "item_id", "Unnamed: 0"]
                ]
                if quarter_cols:
                    latest_quarter = quarter_cols[0]
                    for _, row in ratio_df.iterrows():
                        item_id = row.get("item_id")
                        if pd.notna(item_id):
                            ratio_map[str(item_id)] = row.get(latest_quarter)
            else:
                if isinstance(ratio_df.columns, pd.MultiIndex):
                    work_df = ratio_df.copy()

                    year_col = ("Meta", "yearReport")
                    len_col = ("Meta", "lengthReport")
                    if year_col in work_df.columns:
                        work_df[year_col] = pd.to_numeric(work_df[year_col], errors="coerce")
                    if len_col in work_df.columns:
                        work_df[len_col] = pd.to_numeric(work_df[len_col], errors="coerce")

                    sort_cols = []
                    if year_col in work_df.columns:
                        sort_cols.append(year_col)
                    if len_col in work_df.columns:
                        sort_cols.append(len_col)

                    if sort_cols:
                        work_df = work_df.sort_values(by=sort_cols, ascending=False)

                    latest_row = work_df.iloc[0]

                    if year_col in work_df.columns:
                        year_val = latest_row.get(year_col)
                        if pd.notna(year_val):
                            latest_quarter = str(int(year_val))

                    vci_key_map = {
                        "trailing_eps": ("Chỉ tiêu định giá", "EPS (VND)"),
                        "book_value_per_share_bvps": ("Chỉ tiêu định giá", "BVPS (VND)"),
                        "p_e": ("Chỉ tiêu định giá", "P/E"),
                        "p_b": ("Chỉ tiêu định giá", "P/B"),
                        "p_s": ("Chỉ tiêu định giá", "P/S"),
                        "dividend_yield": ("Chỉ tiêu khả năng sinh lợi", "Dividend yield (%)"),
                        "ev_ebitda": ("Chỉ tiêu định giá", "EV/EBITDA"),
                        "gross_profit_margin": ("Chỉ tiêu khả năng sinh lợi", "Gross Profit Margin (%)"),
                        "net_profit_margin": ("Chỉ tiêu khả năng sinh lợi", "Net Profit Margin (%)"),
                        "roe": ("Chỉ tiêu khả năng sinh lợi", "ROE (%)"),
                        "roa": ("Chỉ tiêu khả năng sinh lợi", "ROA (%)"),
                    }

                    for canonical_key, vci_col in vci_key_map.items():
                        if vci_col in work_df.columns:
                            ratio_map[canonical_key] = latest_row.get(vci_col)

        def clean(value):
            if value is None or (isinstance(value, float) and pd.isna(value)):
                return None
            if isinstance(value, str):
                value = value.replace("\r", " ").replace("\n", " ").strip()
                if value == "":
                    return None
            return value

        lines = []
        
        overview_fields = [
            ("company_name", "Tên công ty"),
            ("symbol", "Mã"),
            ("exchange", "Sàn"),
            ("founded_date", "Ngày thành lập"),
            ("charter_capital", "Vốn điều lệ (Tỉ VNĐ)"),
            ("listed_volume", "Cổ phiếu niêm yết"),
            ("outstanding_shares", "Cổ phiếu lưu hành"),
            ("business_model", "Mô hình kinh doanh"),
            ("email", "Email"),
            ("website", "Website"),
        ]

        for key, label in overview_fields:
            value = clean(overview.get(key))
            if value is not None:
                lines.append(f"- {label}: {value}")

        ratio_fields = [
            ("trailing_eps", "EPS 4 quý gần nhất"),
            ("book_value_per_share_bvps", "BVPS"),
            ("p_e", "P/E"),
            ("p_b", "P/B"),
            ("p_s", "P/S"),
            ("dividend_yield", "Tỷ suất cổ tức"),
            ("beta", "Beta"),
            ("ev_ebit", "EV/EBIT"),
            ("ev_ebitda", "EV/EBITDA"),
            ("gross_profit_margin", "Biên LN gộp"),
            ("net_profit_margin", "Biên LN ròng"),
            ("roe", "ROE"),
            ("roa", "ROA"),
        ]

        for key, label in ratio_fields:
            value = clean(ratio_map.get(key))
            if value is not None:
                suffix = f" ({latest_quarter})" if latest_quarter else ""
                lines.append(f"- {label}{suffix}: {value}")

        if len(lines) == 0:
            return f"Không có dữ liệu cơ bản cho mã '{ticker}'"

        header = f"# Dữ liệu cơ bản cho {ticker}\n"
        header += f"# Dữ liệu được lấy vào lúc: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        return header + "\n".join(lines)
    except Exception as e:
        return f"Lỗi khi lấy thông tin công ty cho {symbol}: {str(e)}"


def _clean_news_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _pick_news_timestamp(df: pd.DataFrame) -> pd.Series:
    # Prefer public_date, then created_at, then updated_at; values are usually epoch milliseconds.
    for col in ["public_date", "created_at", "updated_at"]:
        if col not in df.columns:
            continue

        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().any():
            ts = pd.to_datetime(numeric, unit="ms", errors="coerce")
            if ts.notna().any():
                return ts

        ts = pd.to_datetime(df[col], errors="coerce")
        if ts.notna().any():
            return ts

    return pd.to_datetime(pd.Series([None] * len(df)), errors="coerce")


def get_news(
    ticker: Annotated[str, "ticker symbol of the company"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    """Get company news from vnstock VCI source and filter by date range."""
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError:
        return "Định dạng ngày không hợp lệ. Vui lòng dùng yyyy-mm-dd."

    if start_dt > end_dt:
        start_dt, end_dt = end_dt, start_dt

    # Enforce a minimum query window of 1 month.
    min_start_dt = end_dt - relativedelta(months=1)
    if start_dt > min_start_dt:
        start_dt = min_start_dt

    start_date = start_dt.strftime("%Y-%m-%d")
    end_date = end_dt.strftime("%Y-%m-%d")

    symbol = ticker.upper().strip()

    try:
        news_df = Company(symbol=symbol, source="VCI").news()
    except Exception as e:
        return f"Lỗi khi lấy tin tức vnstock cho {symbol}: {str(e)}"

    if news_df is None or news_df.empty:
        return f"Không có tin tức cho mã '{symbol}' trong nguồn VCI."

    df = news_df.copy()
    df["_ts"] = _pick_news_timestamp(df)

    # Keep rows with valid timestamp and filter by inclusive date range.
    df = df[df["_ts"].notna()].copy()
    df = df[(df["_ts"] >= start_dt) & (df["_ts"] <= end_dt + pd.Timedelta(days=1))]

    if df.empty:
        return f"Không có tin tức cho mã '{symbol}' từ {start_date} đến {end_date}."

    df = df.sort_values("_ts", ascending=False)

    lines = []
    for _, row in df.head(30).iterrows():
        title = _clean_news_text(row.get("news_title", ""))
        sub_title = _clean_news_text(row.get("news_sub_title", ""))
        short_content = _clean_news_text(row.get("news_short_content", ""))
        full_content = _clean_news_text(row.get("news_full_content", ""))
        link = _clean_news_text(row.get("news_source_link", ""))
        pub_time = row.get("_ts")
        pub_str = pub_time.strftime("%Y-%m-%d %H:%M:%S") if pd.notna(pub_time) else "N/A"

        summary = sub_title or short_content or full_content
        if len(summary) > 600:
            summary = summary[:600].rstrip() + "..."

        lines.append(f"### {title if title else 'Không có tiêu đề'}")
        lines.append(f"- Thời gian: {pub_str}")
        if link:
            lines.append(f"- Nguồn: {link}")
        if summary:
            lines.append(f"- Tóm tắt: {summary}")
        lines.append("")

    header = f"## Tin tức cho {symbol}, từ {start_date} đến {end_date}\n"
    header += f"# Tổng số tin đã lọc: {len(df)}\n"
    header += f"# Dữ liệu được lấy vào lúc: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    return header + "\n".join(lines)
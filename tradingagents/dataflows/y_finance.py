from typing import Annotated, Optional
from datetime import datetime
from dateutil.relativedelta import relativedelta
import yfinance as yf
import pandas as pd
from .stockstats_utils import StockstatsUtils

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
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
):

    datetime.strptime(start_date, "%Y-%m-%d")
    datetime.strptime(end_date, "%Y-%m-%d")

    # Create ticker object
    ticker = yf.Ticker(_normalize_vn_ticker(symbol))

    # Fetch historical data for the specified date range
    data = ticker.history(start=start_date, end=end_date)

    # Check if data is empty
    if data.empty:
        return (
            f"Không có dữ liệu cho mã '{symbol}' từ {start_date} đến {end_date}"
        )

    # Remove timezone info from index for cleaner output
    if isinstance(data.index, pd.DatetimeIndex) and data.index.tz is not None:
        data.index = data.index.tz_localize(None)

    # Round numerical values to 2 decimal places for cleaner display
    numeric_columns = ["Open", "High", "Low", "Close", "Adj Close"]
    for col in numeric_columns:
        if col in data.columns:
            data[col] = data[col].round(2)

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
        indicator_data = _get_stock_stats_bulk(symbol, indicator, curr_date)
        
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


def _get_stock_stats_bulk(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicator: Annotated[str, "technical indicator to calculate"],
    curr_date: Annotated[str, "current date for reference"]
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
    curr_date: Annotated[Optional[str], "current date (not used for yfinance)"] = None
):
    """Get balance sheet data from yfinance."""
    try:
        ticker_obj = yf.Ticker(_normalize_vn_ticker(ticker))
        
        if freq.lower() == "quarterly":
            data = ticker_obj.quarterly_balance_sheet
        else:
            data = ticker_obj.balance_sheet
            
        if data.empty:
            return f"Không có dữ liệu bảng cân đối kế toán cho mã '{ticker}'"
            
        # Convert to CSV string for consistency with other functions
        csv_string = data.to_csv()
        
        # Add header information
        freq_vn = "năm" if freq == "year" else "quý"
        header = f"# Dữ liệu bảng cân đối kế toán cho {ticker.upper()} ({freq_vn})\n"
        header += f"# Dữ liệu được lấy vào lúc: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        return header + csv_string
        
    except Exception as e:
        return f"Lỗi khi lấy bảng cân đối kế toán cho {ticker}: {str(e)}"


def get_cashflow(
    ticker: Annotated[str, "ticker symbol of the company"],
    freq: Annotated[str, "frequency of data: 'annual' or 'quarterly'"] = "quarterly",
    curr_date: Annotated[Optional[str], "current date (not used for yfinance)"] = None
):
    """Get cash flow data from yfinance."""
    try:
        ticker_obj = yf.Ticker(_normalize_vn_ticker(ticker))
        
        if freq.lower() == "quarterly":
            data = ticker_obj.quarterly_cashflow
        else:
            data = ticker_obj.cashflow
            
        if data.empty:
            return f"Không có dữ liệu lưu chuyển tiền tệ cho mã '{ticker}'"
            
        # Convert to CSV string for consistency with other functions
        csv_string = data.to_csv()
        
        # Add header information
        header = f"# Dữ liệu lưu chuyển tiền tệ cho {ticker.upper()} ({freq})\n"
        header += f"# Dữ liệu được lấy vào lúc: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        return header + csv_string
        
    except Exception as e:
        return f"Lỗi khi lấy báo cáo lưu chuyển tiền tệ cho {ticker}: {str(e)}"


def get_income_statement(
    ticker: Annotated[str, "ticker symbol of the company"],
    freq: Annotated[str, "frequency of data: 'annual' or 'quarterly'"] = "quarterly",
    curr_date: Annotated[Optional[str], "current date (not used for yfinance)"] = None
):
    """Get income statement data from yfinance."""
    try:
        ticker_obj = yf.Ticker(_normalize_vn_ticker(ticker))
        
        if freq.lower() == "quarterly":
            data = ticker_obj.quarterly_income_stmt
        else:
            data = ticker_obj.income_stmt
            
        if data.empty:
            return f"Không có dữ liệu báo cáo kết quả kinh doanh cho mã '{ticker}'"
            
        # Convert to CSV string for consistency with other functions
        csv_string = data.to_csv()
        
        # Add header information
        header = f"# Dữ liệu báo cáo kết quả kinh doanh cho {ticker.upper()} ({freq})\n"
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
    curr_date: Annotated[Optional[str], "current date (not used for yfinance)"] = None,
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
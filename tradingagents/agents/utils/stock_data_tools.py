from langchain_core.tools import tool
from typing import Annotated
from tradingagents.dataflows.interface import route_to_vendor


@tool
def get_stock_data(
    symbol: Annotated[str, "ticker symbol of the company"],
    curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
    look_back_days: Annotated[int, "Number of days to look back"],
) -> str:
    """
    Retrieve stock price data (OHLCV) for a given ticker symbol.
    Uses the configured core_stock_apis vendor.
    Args:
        symbol (str): Ticker symbol of the company, e.g. AAPL, TSM
        curr_date (str): Current date in yyyy-mm-dd format
        look_back_days (int): Number of days to look back
    Returns:
        str: A formatted dataframe containing stock price data for the look-back window ending at curr_date.
    """
    return route_to_vendor("get_stock_data", symbol, curr_date, look_back_days)

@tool
def get_indicators(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicator: Annotated[str, "technical indicator to get the analysis and report of"],
    curr_date: Annotated[str, "The current trading date you are trading on, YYYY-mm-dd"],
    look_back_days: Annotated[int, "how many days to look back"] = 30,
) -> str:
    """
    Retrieve technical indicators for a given ticker symbol.
    Uses the configured technical_indicators vendor.
    Args:
        symbol (str): Ticker symbol of the company, e.g. AAPL, TSM
        indicator (str): Technical indicator to get the analysis and report of
        curr_date (str): The current trading date you are trading on, YYYY-mm-dd
        look_back_days (int): How many days to look back, default is 30
    Returns:
        str: A formatted dataframe containing the technical indicators for the specified ticker symbol and indicator.
    """
    return route_to_vendor("get_indicators", symbol, indicator, curr_date, look_back_days)

@tool
def get_market_context(
    ticker: Annotated[str, "ticker symbol đang phân tích"],
    curr_date: Annotated[str, "ngày tham chiếu YYYY-mm-dd"],
) -> str:
    """Retrieve market trend and breadth context"""
    return route_to_vendor("get_market_context", ticker, curr_date)

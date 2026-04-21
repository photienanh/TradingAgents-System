from langchain_core.tools import tool
from typing import Annotated
from tradingagents.dataflows.interface import route_to_vendor

@tool
def get_f247_forum_posts(
    ticker: Annotated[str, "Ticker symbol"],
    curr_date: Annotated[str, "Current date you are trading at, yyyy-mm-dd format"],
    look_back_days: Annotated[int, "Number of days to look back"] = 30,
    max_threads: Annotated[int, "Maximum number of threads to include"] = 10,
    max_posts_per_thread: Annotated[int, "Maximum number of posts per thread"] = 10,
) -> str:
    """
    Retrieve forum discussion posts for a ticker from F247.
    Uses the configured social_data vendor.
    """
    return route_to_vendor(
        "get_f247_forum_posts",
        ticker,
        curr_date,
        look_back_days,
        max_threads,
        max_posts_per_thread,
    )


@tool
def get_ticker_news(
    ticker: Annotated[str, "Ticker symbol"],
    curr_date: Annotated[str, "Current date you are trading at, yyyy-mm-dd format"],
    look_back_days: Annotated[int, "Number of days to look back"] = 30,
    max_items: Annotated[int, "Maximum number of articles to return"] = 10,
) -> str:
    """
    Retrieve ticker-specific news from RSS/news sources.
    Uses the configured social_data vendor.
    """
    return route_to_vendor("get_ticker_news", ticker, curr_date, look_back_days, max_items)

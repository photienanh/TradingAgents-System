import os

DEFAULT_CONFIG = {
    "project_dir": os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
    "results_dir": os.getenv("TRADINGAGENTS_RESULTS_DIR", "./results"),
    "data_cache_dir": os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
        "dataflows/data_cache",
    ),
    # LLM settings
    "llm_provider": "openai",
    "deep_think_llm": "o4-mini",
    "quick_think_llm": "gpt-4o-mini",
    "backend_url": "https://api.openai.com/v1",
    # Debate and discussion settings
    "max_debate_rounds": 1,
    "max_risk_discuss_rounds": 1,
    "max_recur_limit": 100,
    # Data vendor configuration
    # Category-level configuration (default for all tools in category)
    "data_vendors": {
        "core_stock_apis": "vnstock",       # Options: yfinance, vnstock
        "technical_indicators": "vnstock",  # Options: yfinance, vnstock
        "fundamental_data": "vnstock", # Options: openai, vnstock, yfinance
        "news_data": "google",        # Options: openai, google, vnstock
        "global_data": "vietstock",      # Options: openai, vietstock
        "insider_transaction_data": "yfinance", # Options: yfinance
    },
    # Tool-level configuration (takes precedence over category-level)
    "tool_vendors": {
        # Example: "get_stock_data": "yfinance",       # Override category default
        # Example: "get_news": "openai",               # Override category default
        "get_news": "google,vnstock",                  # Google first, fallback to vnstock
    },
}

"""
tradingagents/default_config.py
Default runtime configuration for TradingAgents.
"""

import os

# ── Đường dẫn đến project AlphaGPT ───────────────────────────────────
# Có thể đặt ALPHAGPT_ROOT để trỏ sang repo AlphaGPT khác.
# Mặc định dùng root của repo TradingAgents hiện tại (chứa thư mục data/).
_ALPHAGPT_ROOT = os.getenv(
    "ALPHAGPT_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
)

DEFAULT_CONFIG = {
    "project_dir": os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
    "results_dir": os.getenv("TRADINGAGENTS_RESULTS_DIR", "./results"),
    "data_cache_dir": os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
        "dataflows/data_cache",
    ),

    # LLM
    "llm_provider":   "openai",
    "deep_think_llm": "o4-mini",
    "quick_think_llm":"gpt-4o-mini",
    "fallback_llm":    "openai/gpt-oss-120b",
    "backend_url":    "https://api.openai.com/v1",
    "fallback_url": "https://api.groq.com/openai/v1",

    # Debate
    "max_debate_rounds":       1,
    "max_risk_discuss_rounds": 1,
    "max_recur_limit":         100,

    # New AlphaGPT inputs:
    # - global alpha library at data/alpha_library.json
    # - per-ticker market data at data/market_data/*.csv
    "alpha_library_path": os.path.join(_ALPHAGPT_ROOT, "data", "alpha_library.json"),
    "market_data_dir": os.path.join(_ALPHAGPT_ROOT, "data", "market_data"),

    # Backward-compatible keys still consumed by some graph wiring.
    "alpha_formula_dir": os.path.join(_ALPHAGPT_ROOT, "data", "alpha_library.json"),
    "alpha_values_dir": os.path.join(_ALPHAGPT_ROOT, "data", "market_data"),

    # Data vendors
    "data_vendors": {
        "core_stock_apis":        "vnstock",
        "technical_indicators":   "vnstock",
        "fundamental_data":       "yfinance",
        "news_data":              "cafef",
        "social_data":            "f247",
        "global_data":            "vietstock",
        "insider_transaction_data":"yfinance",
    },
    "tool_vendors": {
        "get_news": "cafef, vnstock",
    },
}
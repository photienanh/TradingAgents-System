"""
tradingagents/default_config.py
Default runtime configuration for TradingAgents.
"""

import os

DEFAULT_CONFIG = {
    "project_dir": os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
    "results_dir": os.getenv("TRADINGAGENTS_RESULTS_DIR", "./results"),


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
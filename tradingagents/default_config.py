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

    # TradingAgents đọc từ các file này để tạo quant_report.
    # Cần chạy AlphaGPT pipeline trước (pipelines/gen_alpha.py) để có data.
    "alpha_formula_dir": os.path.join(_ALPHAGPT_ROOT, "data", "alpha_formulas"),
    "alpha_values_dir":  os.path.join(_ALPHAGPT_ROOT, "data", "alphas"),

    # Data vendors
    "data_vendors": {
        "core_stock_apis":        "vnstock",
        "technical_indicators":   "vnstock",
        "fundamental_data":       "yfinance",
        "news_data":              "google",
        "global_data":            "vietstock",
        "insider_transaction_data":"yfinance",
    },
    "tool_vendors": {
        "get_news": "google, vnstock",
    },
}
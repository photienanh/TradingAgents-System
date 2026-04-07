"""
alpha/core/paths.py
BASE_DIR = TradingAgents root (cha của alpha/, tradingagents/, app/, data/)
"""
from pathlib import Path

BASE_DIR  = Path(__file__).resolve().parents[2]   # TradingAgents/
DATA_DIR  = BASE_DIR / "data"

ALPHA_FORMULA_DIR    = str(DATA_DIR / "alpha_formulas")
ALPHA_VALUES_DIR     = str(DATA_DIR / "alphas")
ALPHA_MEMORY_DIR     = str(DATA_DIR / "alpha_memory")
FEATURES_DIR         = str(DATA_DIR / "features")
PRICE_DIR            = str(DATA_DIR / "price")
RAW_NEWS_DIR         = str(DATA_DIR / "raw_news")
DAILY_SCORES_DIR     = str(DATA_DIR / "daily_scores")
SENTIMENT_OUTPUT_DIR = str(DATA_DIR / "sentiment_output")
SIGNALS_DIR          = str(DATA_DIR / "signals")

for _d in [ALPHA_FORMULA_DIR, ALPHA_VALUES_DIR, ALPHA_MEMORY_DIR,
           FEATURES_DIR, PRICE_DIR, RAW_NEWS_DIR,
           DAILY_SCORES_DIR, SENTIMENT_OUTPUT_DIR, SIGNALS_DIR]:
    Path(_d).mkdir(parents=True, exist_ok=True)
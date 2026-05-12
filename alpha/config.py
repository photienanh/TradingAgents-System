"""
config.py
Tập trung tất cả hyperparameters và thresholds của pipeline.
"""
from dataclasses import dataclass


@dataclass
class PipelineConfig:
    # Backtest
    ic_signal_threshold: float = 0.02
    sharpe_min_threshold: float = 1.0
    return_min_threshold: float = 0.0
    test_ratio: float = 0.3

    # Sota selection
    min_sota: int = 3
    max_sota: int = 5
    corr_threshold: float = 0.55

    # GP
    gp_iterations: int = 15
    population_size: int = 6

    # RAG
    rag_top_k: int = 5

    # Data
    min_history_days: int = 30
    forward_return_horizon: int = 2  # T+2 settlement (changed from T+1)
    
    # Costs
    short_tax_rate: float = 0.001  # 0.1% VN transaction tax on short positions
    cost_per_turnover: float = 0.0015  # 15bps per unit turnover


DEFAULT_CONFIG = PipelineConfig()
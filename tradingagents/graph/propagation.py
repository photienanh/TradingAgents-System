"""
tradingagents/graph/propagation.py
State propagation helpers for graph execution.
"""

from typing import Dict, Any
from tradingagents.agents.utils.agent_states import AgentState, InvestDebateState, RiskDebateState


class Propagator:
    def __init__(self, max_recur_limit=100):
        self.max_recur_limit = max_recur_limit

    def create_initial_state(
        self,
        company_name: str,
        trade_date: str,
        alpha_signal: Dict[str, Any] | None = None,
        trading_horizon: str = "short",
    ) -> Dict[str, Any]:
        return {
            "messages":           [("human", company_name)],
            "company_of_interest": company_name,
            "trade_date":         str(trade_date),
            "trading_horizon":    trading_horizon,

            # ── Analyst reports ────────────────────────────────────────
            "market_report":       "",
            "sentiment_report":    "",
            "news_report":         "",
            "fundamentals_report": "",
            "quant_report":        "",

            # ── Researcher debate ──────────────────────────────────────
            "investment_debate_state": InvestDebateState({
                "bull_history":     "",
                "bear_history":     "",
                "history":          "",
                "current_response": "",
                "judge_decision":   "",
                "count":            0,
            }),
            "investment_plan": "",

            # ── Risk debate ────────────────────────────────────────────
            "risk_debate_state": RiskDebateState({
                "history":                    "",
                "risky_history":              "",
                "safe_history":               "",
                "neutral_history":            "",
                "latest_speaker":             "",
                "current_risky_response":     "",
                "current_safe_response":      "",
                "current_neutral_response":   "",
                "judge_decision":             "",
                "count":                      0,
            }),
            "final_trade_decision": "",
        }

    def get_graph_args(self) -> Dict[str, Any]:
        return {
            "stream_mode": "values",
            "config":      {"recursion_limit": self.max_recur_limit},
        }
"""
tradingagents/graph/trading_graph.py
Graph orchestration for TradingAgents.
"""

import os
from pathlib import Path
import json
import asyncio
from datetime import date
from typing import Dict, Any, Tuple, Optional, cast, Callable
import logging

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode
from langchain_core.runnables import Runnable as _LCRunnable

from tradingagents.agents import *
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.agents.utils.memory import FinancialSituationMemory
from tradingagents.agents.utils.agent_states import AgentState, InvestDebateState, RiskDebateState
from tradingagents.dataflows.config import set_config
from tradingagents.agents.utils.agent_utils import (
    get_stock_data, get_indicators, get_market_context, get_fundamentals,
    get_balance_sheet, get_cashflow, get_income_statement,
    get_news, get_insider_transactions, get_global_news,
)

from .conditional_logic import ConditionalLogic
from .setup import GraphSetup
from .propagation import Propagator
from .reflection import Reflector
from .signal_processing import SignalProcessor


logger = logging.getLogger(__name__)


class ResilientChatModel(_LCRunnable):
    """Wrapper that retries calls on a fallback chat model when primary fails."""

    def __init__(self, primary_model, fallback_model=None, label: str = "llm"):
        self._primary = primary_model
        self._fallback = fallback_model
        self._label = label

    def __or__(self, other):
        return self._primary.__or__(other)

    def __ror__(self, other):
        return self._primary.__ror__(other)
    
    def invoke(self, input, config=None, **kwargs):
        try:
            return self._primary.invoke(input, config=config, **kwargs)
        except Exception as exc:
            if self._fallback is None:
                raise
            logger.warning(...)
            return self._fallback.invoke(input, config=config, **kwargs)

    async def ainvoke(self, *args, **kwargs):
        try:
            return await self._primary.ainvoke(*args, **kwargs)
        except Exception as exc:
            if self._fallback is None:
                raise
            logger.warning("[%s] Primary model failed (async), fallback will be used: %s", self._label, exc)
            return await self._fallback.ainvoke(*args, **kwargs)

    def bind_tools(self, tools):
        primary_bound = self._primary.bind_tools(tools)
        fallback_bound = self._fallback.bind_tools(tools) if self._fallback is not None else None
        return ResilientChatModel(primary_bound, fallback_bound, label=self._label)

    def __getattr__(self, attr):
        return getattr(self._primary, attr)


class TradingAgentsGraph:
    def __init__(
        self,
        selected_analysts=["market", "social", "news", "fundamentals"],
        debug=False,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.debug  = debug
        self.config = config or DEFAULT_CONFIG
        set_config(self.config)

        os.makedirs(
            os.path.join(self.config["project_dir"], "dataflows/data_cache"),
            exist_ok=True,
        )

        self.deep_thinking_llm = self._build_llm_with_fallback(
            model_name=self.config["deep_think_llm"],
            label="deep_think_llm",
        )
        self.quick_thinking_llm = self._build_llm_with_fallback(
            model_name=self.config["quick_think_llm"],
            label="quick_think_llm",
        )

        self.bull_memory         = FinancialSituationMemory("bull_memory",         self.config)
        self.bear_memory         = FinancialSituationMemory("bear_memory",         self.config)
        self.trader_memory       = FinancialSituationMemory("trader_memory",       self.config)
        self.invest_judge_memory = FinancialSituationMemory("invest_judge_memory", self.config)
        self.risk_manager_memory = FinancialSituationMemory("risk_manager_memory", self.config)

        self.tool_nodes = self._create_tool_nodes()
        self.conditional_logic = ConditionalLogic(
            max_debate_rounds=int(self.config.get("max_debate_rounds", 1) or 1),
            max_risk_discuss_rounds=int(self.config.get("max_risk_discuss_rounds", 1) or 1),
        )

        alpha_formula_dir = self.config.get("alpha_formula_dir", "")
        alpha_values_dir  = self.config.get("alpha_values_dir",  "")

        self.graph_setup = GraphSetup(
            self.quick_thinking_llm,
            self.deep_thinking_llm,
            self.tool_nodes,
            self.bull_memory,
            self.bear_memory,
            self.trader_memory,
            self.invest_judge_memory,
            self.risk_manager_memory,
            self.conditional_logic,
            alpha_formula_dir=alpha_formula_dir,
            alpha_values_dir=alpha_values_dir,
        )

        self.propagator      = Propagator()
        self.reflector       = Reflector(self.quick_thinking_llm)
        self.signal_processor = SignalProcessor(self.quick_thinking_llm)

        self.curr_state       = None
        self.ticker           = None
        self.log_states_dict  = {}

        self.graph = self.graph_setup.setup_graph(selected_analysts)

    @staticmethod
    def _normalize_alphagpt_signal(alphagpt_signal: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Chuẩn hoá alpha signal cho graph state:
        - Dùng strict ic_oos (không fallback từ avg_ic)
        - Chuẩn hoá kiểu dữ liệu số nếu có
        """
        if not isinstance(alphagpt_signal, dict):
            return {}

        normalized = dict(alphagpt_signal)

        def _to_float(v: Any) -> Optional[float]:
            try:
                return float(v) if v is not None else None
            except (TypeError, ValueError):
                return None

        normalized["ic_oos"] = _to_float(normalized.get("ic_oos"))
        if "avg_ic" in normalized:
            normalized["avg_ic"] = _to_float(normalized.get("avg_ic"))
        return normalized

    def _build_llm_with_fallback(self, model_name: str, label: str) -> ResilientChatModel:
        backend_url = self.config.get("backend_url", "https://api.openai.com/v1")
        fallback_model = self.config.get("fallback_llm", "openai/gpt-oss-120b")
        fallback_url = self.config.get("fallback_url", "https://api.groq.com/openai/v1")

        openai_key = os.getenv("OPENAI_API_KEY")
        groq_key = os.getenv("GROQ_API_KEY")

        def _is_groq_model(name: str) -> bool:
            return str(name).strip().lower() == "openai/gpt-oss-120b"

        if _is_groq_model(model_name):
            if not groq_key:
                raise RuntimeError("GROQ_API_KEY is required when using openai/gpt-oss-120b")
            primary = ChatOpenAI(model=model_name, base_url=fallback_url, api_key=groq_key)
        else:
            primary = ChatOpenAI(model=model_name, base_url=backend_url, api_key=openai_key)

        fallback = None
        if fallback_model and str(fallback_model) != str(model_name) and groq_key:
            fallback = ChatOpenAI(model=fallback_model, base_url=fallback_url, api_key=groq_key)
        elif fallback_model and str(fallback_model) != str(model_name) and not groq_key:
            logger.warning("[%s] GROQ_API_KEY missing; fallback model is disabled", label)

        if fallback is not None:
            logger.info("[%s] Enabled fallback model %s via Groq", label, fallback_model)

        return ResilientChatModel(primary, fallback, label=label)

    def _create_tool_nodes(self) -> Dict[str, ToolNode]:
        return {
            "market":       ToolNode([get_stock_data, get_indicators, get_market_context]),
            "social":       ToolNode([get_news]),
            "news":         ToolNode([get_news, get_global_news, get_insider_transactions]),
            "fundamentals": ToolNode([get_fundamentals, get_balance_sheet, get_cashflow, get_income_statement]),
        }

    def propagate(
        self,
        company_name,
        trade_date,
        alphagpt_signal: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        self.ticker = company_name
        normalized_alpha_signal = self._normalize_alphagpt_signal(alphagpt_signal)
        init_agent_state = self.propagator.create_initial_state(
            company_name, trade_date, alphagpt_signal=normalized_alpha_signal
        )
        args = self.propagator.get_graph_args()

        should_stream = self.debug or progress_callback is not None
        if should_stream:
            final_state = None
            for chunk in self.graph.stream(cast(AgentState, init_agent_state), **args):
                final_state = chunk
                if progress_callback is not None:
                    try:
                        progress_callback(chunk)
                    except Exception as exc:
                        if isinstance(exc, asyncio.CancelledError):
                            raise
            if final_state is None:
                raise RuntimeError("Graph stream returned no state")
        else:
            final_state = self.graph.invoke(cast(AgentState, init_agent_state), **args)

        self.curr_state = final_state
        self._log_state(trade_date, final_state)
        return final_state, self.process_signal(final_state["final_trade_decision"])

    def _log_state(self, trade_date, final_state):
        self.log_states_dict[str(trade_date)] = {
            "company_of_interest":    final_state["company_of_interest"],
            "trade_date":             final_state["trade_date"],
            "market_report":          final_state["market_report"],
            "sentiment_report":       final_state["sentiment_report"],
            "news_report":            final_state["news_report"],
            "fundamentals_report":    final_state["fundamentals_report"],
            "quant_report":           final_state.get("quant_report", ""),
            "investment_debate_state": {
                "bull_history":    final_state["investment_debate_state"]["bull_history"],
                "bear_history":    final_state["investment_debate_state"]["bear_history"],
                "history":         final_state["investment_debate_state"]["history"],
                "current_response":final_state["investment_debate_state"]["current_response"],
                "judge_decision":  final_state["investment_debate_state"]["judge_decision"],
            },
            "trader_investment_decision": final_state["trader_investment_plan"],
            "risk_debate_state": {
                "risky_history":   final_state["risk_debate_state"]["risky_history"],
                "safe_history":    final_state["risk_debate_state"]["safe_history"],
                "neutral_history": final_state["risk_debate_state"]["neutral_history"],
                "history":         final_state["risk_debate_state"]["history"],
                "judge_decision":  final_state["risk_debate_state"]["judge_decision"],
            },
            "investment_plan":       final_state["investment_plan"],
            "final_trade_decision":  final_state["final_trade_decision"],
        }

        directory = Path(f"eval_results/{self.ticker}/")
        directory.mkdir(parents=True, exist_ok=True)
        with open(f"eval_results/{self.ticker}/full_states_log_{trade_date}.json", "w", encoding="utf-8") as f:
            json.dump(self.log_states_dict, f, indent=4, ensure_ascii=False)

    def reflect_and_remember(self, returns_losses):
        self.reflector.reflect_bull_researcher(self.curr_state, returns_losses, self.bull_memory)
        self.reflector.reflect_bear_researcher(self.curr_state, returns_losses, self.bear_memory)
        self.reflector.reflect_trader(self.curr_state, returns_losses, self.trader_memory)
        self.reflector.reflect_invest_judge(self.curr_state, returns_losses, self.invest_judge_memory)
        self.reflector.reflect_risk_manager(self.curr_state, returns_losses, self.risk_manager_memory)

    def process_signal(self, full_signal):
        return self.signal_processor.process_signal(full_signal)
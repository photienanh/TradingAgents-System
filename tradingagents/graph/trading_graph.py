"""
tradingagents/graph/trading_graph.py
"""

import os
import asyncio
from datetime import date
from typing import Dict, Any, Tuple, Optional, cast, Callable
import logging

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode
from langchain_core.runnables import Runnable as _LCRunnable

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.agents.utils.agent_states import AgentState, InvestDebateState, RiskDebateState
from tradingagents.dataflows.config import set_config
from tradingagents.agents.utils.agent_utils import (
    get_stock_data, get_indicators, get_market_context, get_fundamentals,
    get_balance_sheet, get_cashflow, get_income_statement,
    get_news, get_insider_transactions, get_global_news,
    get_f247_forum_posts, get_ticker_news,
)

from tradingagents.graph.conditional_logic import ConditionalLogic
from tradingagents.graph.setup import GraphSetup
from tradingagents.graph.propagation import Propagator
from tradingagents.graph.signal_processing import SignalProcessor


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

        self.deep_thinking_llm = self._build_llm_with_fallback(
            model_name=self.config["deep_think_llm"],
            label="deep_think_llm",
        )
        self.quick_thinking_llm = self._build_llm_with_fallback(
            model_name=self.config["quick_think_llm"],
            label="quick_think_llm",
        )


        self.tool_nodes = self._create_tool_nodes()
        self.conditional_logic = ConditionalLogic(
            max_debate_rounds=int(self.config.get("max_debate_rounds", 1) or 1),
            max_risk_discuss_rounds=int(self.config.get("max_risk_discuss_rounds", 1) or 1),
        )

        self.graph_setup = GraphSetup(
            self.quick_thinking_llm,
            self.deep_thinking_llm,
            self.tool_nodes,
            self.conditional_logic,
        )

        self.propagator      = Propagator()
        self.signal_processor = SignalProcessor()

        self.ticker           = None

        self.graph = self.graph_setup.setup_graph(selected_analysts)

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
            "social":       ToolNode([get_f247_forum_posts, get_ticker_news]),
            "news":         ToolNode([get_news, get_global_news, get_insider_transactions]),
            "fundamentals": ToolNode([get_fundamentals, get_balance_sheet, get_cashflow, get_income_statement]),
        }

    def propagate(
        self,
        company_name,
        trade_date,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        trading_horizon: str = "short",
    ):
        self.ticker = company_name
        init_agent_state = self.propagator.create_initial_state(
            company_name,
            trade_date,
            trading_horizon=trading_horizon,
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

        return final_state, self.process_signal(final_state["final_trade_decision"])

    def process_signal(self, full_signal):
        return self.signal_processor.process_signal(full_signal)

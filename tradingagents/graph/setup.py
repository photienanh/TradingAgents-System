"""
tradingagents/graph/setup.py
Wires AlphaGPT Analyst vào Tầng 1 (Analyst team).

Thứ tự chạy trong tầng 1:
  Market → Social → News → Fundamentals → AlphaGPT Analyst → Bull Researcher

AlphaGPT Analyst KHÔNG dùng tool call và KHÔNG gọi LLM —
nó chỉ đọc file JSON/CSV đã có sẵn và render report.
Do đó không cần ToolNode và không cần msg_delete node.
"""

from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode

from tradingagents.agents import (
    create_msg_delete,
    create_bull_researcher,
    create_bear_researcher,
    create_research_manager,
    create_fundamentals_analyst,
    create_market_analyst,
    create_neutral_debator,
    create_news_analyst,
    create_risky_debator,
    create_risk_manager,
    create_safe_debator,
    create_social_media_analyst,
    create_trader,
)
from tradingagents.agents.analysts.alphagpt_analyst import create_alphagpt_analyst

from tradingagents.agents.utils.agent_states import AgentState
from .conditional_logic import ConditionalLogic


class GraphSetup:
    def __init__(
        self,
        quick_thinking_llm: ChatOpenAI,
        deep_thinking_llm: ChatOpenAI,
        tool_nodes: Dict[str, ToolNode],
        bull_memory,
        bear_memory,
        trader_memory,
        invest_judge_memory,
        risk_manager_memory,
        conditional_logic: ConditionalLogic,
    ):
        self.quick_thinking_llm  = quick_thinking_llm
        self.deep_thinking_llm   = deep_thinking_llm
        self.tool_nodes          = tool_nodes
        self.bull_memory         = bull_memory
        self.bear_memory         = bear_memory
        self.trader_memory       = trader_memory
        self.invest_judge_memory = invest_judge_memory
        self.risk_manager_memory = risk_manager_memory
        self.conditional_logic   = conditional_logic

    def setup_graph(
        self,
        selected_analysts=["market", "social", "news", "fundamentals"],
    ):
        """
        Dây chuyền analyst:
          [selected_analysts in order] → AlphaGPT Analyst → Bull Researcher → ...

        AlphaGPT Analyst luôn chạy cuối cùng trong tầng analyst,
        sau khi tất cả qualitative analysts đã xong.
        """
        if not selected_analysts:
            raise ValueError("Phải chọn ít nhất một analyst.")

        # ── Tạo nodes ─────────────────────────────────────────────────
        analyst_nodes  = {}
        delete_nodes   = {}
        tool_node_map  = {}

        ANALYST_MAP = {
            "market":       (create_market_analyst,       "Market Analyst"),
            "social":       (create_social_media_analyst, "Social Analyst"),
            "news":         (create_news_analyst,         "News Analyst"),
            "fundamentals": (create_fundamentals_analyst, "Fundamentals Analyst"),
        }

        for key in selected_analysts:
            factory, label = ANALYST_MAP[key]
            analyst_nodes[key] = (factory(self.quick_thinking_llm), label)
            delete_nodes[key]  = create_msg_delete()
            tool_node_map[key] = self.tool_nodes[key]

        # AlphaGPT Analyst node — không cần LLM, không cần tool node
        alphagpt_node          = create_alphagpt_analyst(llm=self.quick_thinking_llm)
        bull_researcher_node   = create_bull_researcher(self.quick_thinking_llm, self.bull_memory)
        bear_researcher_node   = create_bear_researcher(self.quick_thinking_llm, self.bear_memory)
        research_manager_node  = create_research_manager(self.deep_thinking_llm, self.invest_judge_memory)
        trader_node            = create_trader(self.quick_thinking_llm, self.trader_memory)
        risky_analyst          = create_risky_debator(self.quick_thinking_llm)
        neutral_analyst        = create_neutral_debator(self.quick_thinking_llm)
        safe_analyst           = create_safe_debator(self.quick_thinking_llm)
        risk_manager_node      = create_risk_manager(self.deep_thinking_llm, self.risk_manager_memory)

        # ── Build graph ────────────────────────────────────────────────
        workflow = StateGraph(AgentState)

        # Đăng ký analyst nodes
        for key, (node_fn, label) in analyst_nodes.items():
            workflow.add_node(label, node_fn)
            workflow.add_node(f"Msg Clear {label.split()[0]}", delete_nodes[key])
            workflow.add_node(f"tools_{key}", tool_node_map[key])

        # Đăng ký AlphaGPT node (không tool, không delete msg)
        workflow.add_node("AlphaGPT Analyst", alphagpt_node)

        # Đăng ký downstream nodes
        workflow.add_node("Bull Researcher",  bull_researcher_node)
        workflow.add_node("Bear Researcher",  bear_researcher_node)
        workflow.add_node("Research Manager", research_manager_node)
        workflow.add_node("Trader",           trader_node)
        workflow.add_node("Risky Analyst",    risky_analyst)
        workflow.add_node("Neutral Analyst",  neutral_analyst)
        workflow.add_node("Safe Analyst",     safe_analyst)
        workflow.add_node("Risk Judge",       risk_manager_node)

        # ── Edges: Analyst chain ───────────────────────────────────────
        first_key   = selected_analysts[0]
        first_label = analyst_nodes[first_key][1]
        workflow.add_edge(START, first_label)

        COND_MAP = {
            "market":       self.conditional_logic.should_continue_market,
            "social":       self.conditional_logic.should_continue_social,
            "news":         self.conditional_logic.should_continue_news,
            "fundamentals": self.conditional_logic.should_continue_fundamentals,
        }

        for i, key in enumerate(selected_analysts):
            label       = analyst_nodes[key][1]
            prefix      = label.split()[0]
            clear_node  = f"Msg Clear {prefix}"
            tool_node   = f"tools_{key}"

            workflow.add_conditional_edges(
                label,
                COND_MAP[key],
                [tool_node, clear_node],
            )
            workflow.add_edge(tool_node, label)

            # Sau khi clear message: tiến đến analyst tiếp theo
            # hoặc đến AlphaGPT nếu là analyst cuối cùng
            if i < len(selected_analysts) - 1:
                next_label = analyst_nodes[selected_analysts[i + 1]][1]
                workflow.add_edge(clear_node, next_label)
            else:
                # ── Tầng 1 hoàn tất → AlphaGPT Analyst ──────────────
                workflow.add_edge(clear_node, "AlphaGPT Analyst")

        # AlphaGPT Analyst → Bull Researcher (bắt đầu tầng 2)
        workflow.add_edge("AlphaGPT Analyst", "Bull Researcher")

        # ── Edges: Researcher debate ───────────────────────────────────
        workflow.add_conditional_edges(
            "Bull Researcher",
            self.conditional_logic.should_continue_debate,
            {"Bear Researcher": "Bear Researcher", "Research Manager": "Research Manager"},
        )
        workflow.add_conditional_edges(
            "Bear Researcher",
            self.conditional_logic.should_continue_debate,
            {"Bull Researcher": "Bull Researcher", "Research Manager": "Research Manager"},
        )

        # ── Edges: Trader ──────────────────────────────────────────────
        workflow.add_edge("Research Manager", "Trader")

        # ── Edges: Risk debate ─────────────────────────────────────────
        workflow.add_edge("Trader", "Risky Analyst")
        workflow.add_conditional_edges(
            "Risky Analyst",
            self.conditional_logic.should_continue_risk_analysis,
            {"Safe Analyst": "Safe Analyst", "Risk Judge": "Risk Judge"},
        )
        workflow.add_conditional_edges(
            "Safe Analyst",
            self.conditional_logic.should_continue_risk_analysis,
            {"Neutral Analyst": "Neutral Analyst", "Risk Judge": "Risk Judge"},
        )
        workflow.add_conditional_edges(
            "Neutral Analyst",
            self.conditional_logic.should_continue_risk_analysis,
            {"Risky Analyst": "Risky Analyst", "Risk Judge": "Risk Judge"},
        )
        workflow.add_edge("Risk Judge", END)

        return workflow.compile()
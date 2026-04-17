"""
tradingagents/agents/trader/trader.py
Thêm quant_report để Trader có thêm evidence định lượng khi ra plan.
"""

import functools

from tradingagents.agents.utils.text_sanitize import sanitize_for_prompt


def create_trader(llm, memory):
    def trader_node(state, name):
        company_name   = state["company_of_interest"]
        investment_plan = state["investment_plan"]
        market_research_report = state["market_report"]
        sentiment_report       = state["sentiment_report"]
        news_report            = state["news_report"]
        fundamentals_report    = state["fundamentals_report"]
        quant_report = state.get("quant_report", "Không có dữ liệu quant.")

        curr_situation = (
            f"{market_research_report}\n\n{sentiment_report}\n\n"
            f"{news_report}\n\n{fundamentals_report}"
        )
        past_memories = memory.get_memories(curr_situation, n_matches=2)
        past_memory_str = (
            "\n\n".join(r["recommendation"] for r in past_memories)
            if past_memories else "Không có bài học quá khứ phù hợp."
        )

        messages = [
            {
                "role": "system",
                "content": (
                    f"Bạn là trading agent phân tích dữ liệu thị trường để đưa ra "
                    f"quyết định đầu tư. Bạn có hai nguồn thông tin bổ sung cho nhau:\n\n"
                    f"1. QUANT SIGNAL (AlphaGPT): tín hiệu thống kê từ price/volume history, "
                    f"đã validate trên out-of-sample data. IC_OOS và Sharpe_OOS là các chỉ số "
                    f"đáng tin cậy. Hãy sử dụng chúng để calibrate độ tự tin và position size.\n\n"
                    f"2. QUALITATIVE ANALYSIS: fundamental, news, sentiment — cung cấp context "
                    f"mà quant model không thể nắm bắt được.\n\n"
                    f"Dựa trên phân tích tổng hợp, hãy đưa ra một khuyến nghị cụ thể: "
                    f"buy, sell hoặc hold. Phần kết luận phải rõ ràng và bắt buộc kết thúc "
                    f"bằng đúng cú pháp 'FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**'.\n\n"
                    f"Phản tư từ tình huống tương tự: {sanitize_for_prompt(past_memory_str)}"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"## Quant Signal (AlphaGPT)\n{sanitize_for_prompt(quant_report)}\n\n"
                    f"## Kế hoạch từ Research team\n{sanitize_for_prompt(investment_plan)}\n\n"
                    f"Hãy tổng hợp cả hai nguồn để đưa ra trading decision cho {sanitize_for_prompt(company_name)}. "
                    f"Nếu quant signal và qualitative đồng thuận, hãy tự tin hơn. "
                    f"Nếu mâu thuẫn, hãy giải thích rõ tại sao bạn chọn hướng nào."
                ),
            },
        ]

        result = llm.invoke(messages)
        return {
            "messages":             [result],
            "trader_investment_plan": result.content,
            "sender":               name,
        }

    return functools.partial(trader_node, name="Trader")
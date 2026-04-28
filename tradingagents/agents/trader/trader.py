"""
tradingagents/agents/trader/trader.py
"""

import functools

from tradingagents.agents.utils.text_sanitize import sanitize_for_prompt


def create_trader(llm, memory):
    def trader_node(state, name):
        company_name   = state["company_of_interest"]
        trade_date     = state.get("trade_date", "N/A")
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
                    f"Bạn là Trader — người xây dựng kế hoạch giao dịch chi tiết cho {sanitize_for_prompt(company_name)}.\n\n"

                    "## NGUYÊN TẮC TRUNG LẬP (BẮT BUỘC)\n"
                    "BUY, SELL và HOLD đều là các hành động hợp lệ với xác suất ngang nhau.\n"
                    "Nhiệm vụ của bạn là đưa ra kế hoạch giao dịch dựa trên bằng chứng — không phải theo xu hướng hay áp lực.\n\n"

                    "## CÁCH SỬ DỤNG DỮ LIỆU\n"
                    "Bạn có hai nguồn thông tin:\n"
                    "1. **Quant Signal (AlphaGPT)**: Tín hiệu thống kê từ backtest, validate trên OOS data. "
                    "IC và Sharpe là chỉ số định lượng.\n"
                    "2. **Qualitative Analysis**: Fundamental, news, sentiment — cung cấp context mà quant không nắm được.\n\n"
                    "Khi hai nguồn đồng thuận → tự tin hơn.\n"
                    "Khi hai nguồn mâu thuẫn → phân tích nguyên nhân, không tự động theo một bên nào.\n\n"

                    f"Bài học từ tình huống tương tự: {sanitize_for_prompt(past_memory_str)}"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"## Quant Signal (AlphaGPT)\n{sanitize_for_prompt(quant_report)}\n\n"
                    f"## Kế hoạch từ Research team\n{sanitize_for_prompt(investment_plan)}\n\n"

                    f"Hãy xây dựng kế hoạch giao dịch theo format sau:\n\n"

                    f"### Kế Hoạch Giao Dịch — {sanitize_for_prompt(company_name)} — {trade_date}\n\n"

                    "#### Tổng Hợp Tín Hiệu\n"
                    "- **Quant signal**: [hướng, IC, mức độ tin cậy]\n"
                    "- **Research team**: [quyết định và lý do chính]\n"
                    "- **Mức độ đồng thuận**: [đồng thuận / mâu thuẫn, giải thích]\n\n"

                    "#### Đề Xuất Hành Động: **[BUY / SELL / HOLD]**\n\n"

                    "#### Lý Do\n"
                    "[2-3 lý do chính dựa trên bằng chứng từ cả quant và qualitative]\n\n"

                    "#### Thông Số Giao Dịch\n"
                    "- Khung thời gian nắm giữ: [ngắn/trung/dài hạn, ước lượng]\n"
                    "- Mức giá quan tâm: [nếu BUY: vùng mua; nếu SELL: vùng bán; nếu HOLD: mức thoát lỗ]\n"
                    "- Điều kiện thoát: [khi nào nên xem xét đóng vị thế]\n\n"

                    "#### Rủi Ro Cần Theo Dõi\n"
                    "[2-3 rủi ro quan trọng nhất có thể thay đổi kịch bản]"
                ),
            },
        ]

        result = llm.invoke(messages)
        return {
            "messages":               [result],
            "trader_investment_plan": result.content,
            "sender":                 name,
        }

    return functools.partial(trader_node, name="Trader")
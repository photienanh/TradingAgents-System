"""
tradingagents/agents/trader/trader.py
"""

import functools

from tradingagents.agents.utils.text_sanitize import sanitize_for_prompt


def create_trader(llm, memory):
    def trader_node(state, name):
        company_name           = state["company_of_interest"]
        trade_date             = state.get("trade_date", "N/A")
        horizon                = state.get("trading_horizon", "short")
        investment_plan        = state["investment_plan"]
        market_research_report = state["market_report"]
        sentiment_report       = state["sentiment_report"]
        news_report            = state["news_report"]
        fundamentals_report    = state["fundamentals_report"]

        curr_situation = (
            f"{market_research_report}\n\n{sentiment_report}\n\n"
            f"{news_report}\n\n{fundamentals_report}"
        )
        past_memories = memory.get_memories(curr_situation, n_matches=2)
        past_memory_str = (
            "\n\n".join(r["recommendation"] for r in past_memories)
            if past_memories else "Không có bài học quá khứ phù hợp."
        )

        if horizon == "short":
            horizon_sys = (
                "Bạn đang hỗ trợ chiến lược LƯỚT SÓNG NGẮN HẠN (2-5 ngày). "
                "Tập trung vào momentum, timing entry/exit, và quản lý rủi ro chặt chẽ. "
                "BUY, SELL và HOLD đều là hành động hợp lệ với xác suất ngang nhau."
            )
            output_format = (
                f"### Kế Hoạch Giao Dịch — {sanitize_for_prompt(company_name)} — {trade_date}\n\n"

                "#### Tổng Hợp Tín Hiệu\n"
                "- **Quant signal**: [hướng, IC, mức độ tin cậy]\n"
                "- **Research team**: [quyết định và lý do chính]\n"
                "- **Mức độ đồng thuận**: [đồng thuận / mâu thuẫn, giải thích]\n\n"

                "#### Đề Xuất Hành Động: **[BUY / SELL / HOLD]**\n\n"

                "#### Lý Do\n"
                "[2-3 lý do cụ thể, ưu tiên bằng chứng kỹ thuật và momentum]\n\n"

                "#### Thông Số Giao Dịch\n"
                "- Khung nắm giữ dự kiến: [X–Y ngày giao dịch]\n"
                "- Vùng giá vào lệnh: [giá hoặc khoảng giá cụ thể]\n"
                "- Mục tiêu chốt lời (T+2 đến T+5): [mức giá hoặc % kỳ vọng]\n"
                "- Stop-loss: [mức giá hoặc % cắt lỗ tối đa]\n"
                "- Điều kiện hủy kế hoạch sớm: [catalyst nào khiến thoát ngay]\n\n"

                "#### Rủi Ro Cần Theo Dõi\n"
                "[2-3 rủi ro quan trọng nhất trong khung ngắn hạn]"
            )
            quant_report = state.get("quant_report", "")
            quant_block = (
                f"## Quant Signal (AlphaGPT)\n{sanitize_for_prompt(quant_report)}\n\n"
            ) if quant_report else ""
        else:
            horizon_sys = (
                "Bạn đang hỗ trợ chiến lược ĐẦU TƯ DÀI HẠN. "
                "Tập trung vào giá trị doanh nghiệp, luận điểm tăng trưởng, và margin of safety. "
                "Chỉ có hai quyết định: BUY (nếu có luận điểm rõ ràng và định giá hợp lý) hoặc NOT BUY."
            )
            output_format = (
                f"### Kế Hoạch Giao Dịch — {sanitize_for_prompt(company_name)} — {trade_date}\n\n"

                "#### Tổng Hợp Luận Điểm\n"
                "- **Luận điểm tích cực chính**: [1-2 điểm mạnh nhất từ dữ liệu]\n"
                "- **Rủi ro chính**: [1-2 rủi ro đáng lo ngại nhất]\n\n"

                "#### Quyết Định: **[BUY / NOT BUY]**\n\n"

                "#### Lý Do Quyết Định\n"
                "[2-3 lý do cốt lõi — ưu tiên fundamental, định giá, và catalyst dài hạn]\n\n"

                "#### Điều Kiện Theo Dõi\n"
                "- Nếu BUY: các mốc xem xét thoát (định giá đạt mục tiêu, luận điểm thay đổi)\n"
                "- Nếu NOT BUY: điều kiện nào sẽ khiến xem xét lại\n\n"

                "#### Rủi Ro Dài Hạn Cần Theo Dõi\n"
                "[2-3 rủi ro có thể phá vỡ luận điểm đầu tư]"
            )
            quant_block = ""

        messages = [
            {
                "role": "system",
                "content": (
                    f"Bạn là Trader — người xây dựng kế hoạch giao dịch chi tiết cho {sanitize_for_prompt(company_name)}.\n\n"

                    f"## CHIẾN LƯỢC HIỆN TẠI\n{horizon_sys}\n\n"

                    "## CÁCH SỬ DỤNG DỮ LIỆU\n"
                    "Ưu tiên bằng chứng cụ thể từ dữ liệu được cung cấp. "
                    "Khi các nguồn đồng thuận → tự tin hơn. "
                    "Khi mâu thuẫn → phân tích nguyên nhân, không tự động theo một bên nào.\n\n"

                    f"Bài học từ tình huống tương tự: {sanitize_for_prompt(past_memory_str)}"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"{quant_block}"
                    f"## Kế hoạch từ Research team\n{sanitize_for_prompt(investment_plan)}\n\n"
                    f"Hãy xây dựng kế hoạch giao dịch theo format sau:\n\n"
                    f"{output_format}"
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
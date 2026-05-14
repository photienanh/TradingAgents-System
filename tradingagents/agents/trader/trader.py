"""
tradingagents/agents/trader/trader.py
"""

from tradingagents.agents.utils.text_sanitize import sanitize_for_prompt


def create_trader(llm):
    def trader_node(state):
        company_name    = state["company_of_interest"]
        trade_date      = state.get("trade_date", "N/A")
        horizon         = state.get("trading_horizon", "short")
        quant_report    = state.get("quant_report", "")
        investment_plan = state["investment_plan"]
        
        quant_block = (
            f"## Tín hiệu định lượng (Alpha)\n{sanitize_for_prompt(quant_report)}\n\n"
        ) if quant_report and horizon == "short" else ""

        if horizon == "short":
            horizon_sys = (
                "Bạn đang hỗ trợ chiến lược NGẮN HẠN (2–5 ngày). "
                "Ba quyết định hợp lệ: BUY, SELL, HOLD."
            )
            output_format = (
                f"### Kế Hoạch Giao Dịch - {sanitize_for_prompt(company_name)} - {trade_date}\n\n"
                "#### Tổng Hợp Tín Hiệu\n"
                "[Nhận xét về mức độ đồng thuận hay mâu thuẫn giữa các nguồn: "
                "kết quả nghiên cứu, tín hiệu Alpha (nếu có), và các dữ liệu khác]\n\n"
                "#### Quyết Định: **[BUY / SELL / HOLD]**\n\n"
                "#### Lý Do\n"
                "[Lý do dẫn đến quyết định, dựa trên bằng chứng cụ thể từ dữ liệu]"
            )
        else:
            horizon_sys = (
                "Bạn đang hỗ trợ chiến lược ĐẦU TƯ DÀI HẠN. "
                "Hai quyết định hợp lệ: BUY hoặc NOT BUY."
            )
            output_format = (
                f"### Kế Hoạch Giao Dịch - {sanitize_for_prompt(company_name)} - {trade_date}\n\n"
                "#### Tổng Hợp Luận Điểm\n"
                "[Nhận xét về mức độ đồng thuận hay mâu thuẫn giữa các nguồn: "
                "kết quả nghiên cứu, và các dữ liệu khác]\n\n"
                "#### Quyết Định: **[BUY / NOT BUY]**\n\n"
                "#### Lý Do\n"
                "[Lý do dẫn đến quyết định, ưu tiên fundamental và luận điểm dài hạn]"
            )

        messages = [
            {
                "role": "system",
                "content": (
                    f"Bạn là Trader - người xây dựng kế hoạch giao dịch cho {sanitize_for_prompt(company_name)}.\n\n"
                    f"## CHIẾN LƯỢC\n{horizon_sys}\n\n"
                    "Khi các nguồn đồng thuận, hãy tự tin hơn vào quyết định. "
                    "Khi mâu thuẫn, phân tích nguyên nhân thay vì tự động theo một bên."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"{quant_block}"
                    f"## Kết quả từ Research team\n{sanitize_for_prompt(investment_plan)}\n\n"
                    f"Hãy xây dựng kế hoạch giao dịch theo format sau:\n\n"
                    f"{output_format}"
                ),
            },
        ]

        result = llm.invoke(messages)
        return {
            "messages":               [result],
            "trader_investment_plan": result.content,
        }

    return trader_node
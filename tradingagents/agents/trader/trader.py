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
        market_report   = state.get("market_report", "")
        investment_plan = state["investment_plan"]
        
        if horizon == "short":
            horizon_sys = (
                "Bạn đang hỗ trợ chiến lược LƯỚT SÓNG NGẮN HẠN (Swing Trading 2-5 ngày)."
            )
            quant_block = (
                f"## Tín hiệu định lượng (Alpha)\n{sanitize_for_prompt(quant_report)}\n\n"
            ) if quant_report else ""

            output_format = (
                f"### Kế Hoạch Giao Dịch - {sanitize_for_prompt(company_name)} - {trade_date}\n\n"
                "#### 1. Đánh giá vị thế kỹ thuật\n"
                "- **Chỉ thị từ Research Manager:** [Nêu lại quyết định BUY/SELL/HOLD mà Research Manager đã chốt]\n"
                "- **Đánh giá điểm vào (Timing):** [Dựa trên dữ liệu Market và Alpha, đánh giá xem chỉ thị này có nên thực thi ngay lập tức hay phải chờ đợi]\n\n"
                "#### 2. Thông Số Thực Thi\n"
                "- **Khung thời gian vào lệnh (Entry Timeframe):** [VD: Trong phiên hôm nay, Chờ nhịp chỉnh 1-2 phiên tới...]\n"
                "- **Điều kiện kích hoạt (Entry Trigger):** [VD: Giá bứt phá kháng cự X kèm Volume lớn, MACD cắt lên...]\n"
                "- **Vùng giá thực thi (Execution Zone):** [Khoảng giá Mua/Bán cụ thể trích xuất từ hỗ trợ/kháng cự. Nếu HOLD thì không ghi trường này.]\n"
                "- **Mục tiêu chốt lời (Take Profit):** [Chỉ ghi nếu lệnh là MUA]\n"
                "- **Ngưỡng cắt lỗ (Stop Loss):** [Mức giá cắt lỗ để bảo vệ vốn. Chỉ ghi nếu lệnh là MUA]\n"
                "- **Tỷ lệ Rủi ro/Lợi nhuận (Chỉ tính nếu lệnh là MUA]\n"
                "- **Tỷ trọng vốn (Position Size):** [Đề xuất % giải ngân, Nếu HOLD, không ghi trường này]\n\n"
                "#### 3. Căn cứ thiết lập thông số\n"
                "[Phân tích tại sao chọn các mức giá hỗ trợ/kháng cự và tỷ lệ R/R này dựa trên dữ liệu]"
            )
        else:
            horizon_sys = (
                "Bạn đang hỗ trợ chiến lược ĐẦU TƯ DÀI HẠN (Tích sản/Giá trị)."
            )
            quant_block = ""
            output_format = (
                f"### Kế Hoạch Giao Dịch - {sanitize_for_prompt(company_name)} - {trade_date}\n\n"
                "#### 1. Đánh giá vị thế giá\n"
                "- **Chỉ thị từ Research Manager:** [Nêu lại quyết định BUY/ NOT BUY]\n"
                "- **Đánh giá biên an toàn:** [Nhận xét xu hướng giá dài hạn hiện tại có cho điểm gom mua tốt không]\n\n"
                "#### 2. Thông Số Giải Ngân\n"
                "- **Khung thời gian thực thi (Execution Timeframe):** [VD: Rải lệnh gom trong 2-4 tuần tới...]\n"
                "- **Điều kiện kích hoạt (Trigger):** [Điều kiện kích hoạt giao dịch]\n"
                "- **Vùng giá gom mua (Accumulation Zone):** [Khoảng giá thực thi. Chỉ ghi nếu lệnh là MUA]\n"
                "- **Định giá mục tiêu (Target Valuation):** [Mức giá kỳ vọng. Chỉ ghi nếu lệnh là MUA]\n"
                "- **Điểm vô hiệu hoá (Invalidation Point):** [Giá cắt lỗ hoặc điều kiện cơ bản nào bị phá vỡ. Chỉ ghi nếu lệnh là MUA]\n"
                "- **Tỷ trọng vốn (Position Size):** [Mức phân bổ tối đa cho danh mục, VD: Tối đa 20% NAV. Chỉ ghi nếu lệnh là MUA]\n\n"
                "#### 3. Căn cứ thiết lập thông số\n"
                "[Giải thích nguyên nhân chọn vùng giá gom mua dựa trên đồ thị kỹ thuật dài hạn]"
            )

        messages = [
            {
                "role": "system",
                "content": (
                    f"Bạn là Trader - người xây dựng kế hoạch giao dịch cho {sanitize_for_prompt(company_name)}.\n\n"
                    f"## CHIẾN LƯỢC\n{horizon_sys}\n\n"
                    "Bạn KHÔNG ĐƯỢC PHÉP thay đổi quyết định giao dịch. Quyết định ĐÃ ĐƯỢC CHỐT bởi Research Manager."
                    "Bạn PHẢI sử dụng các mức giá thực tế từ 'Dữ liệu Thị trường' (market_report) để thiết lập các thông số vào lệnh."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"{quant_block}"
                    f"## Dữ liệu thị trường và Phân tích kỹ thuật\n{sanitize_for_prompt(market_report)}\n\n"
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
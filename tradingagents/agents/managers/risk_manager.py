"""
tradingagents/agents/managers/risk_manager.py
"""
from tradingagents.agents.utils.text_sanitize import sanitize_for_prompt

def create_risk_manager(llm):
    def risk_manager_node(state) -> dict:

        company_name      = state["company_of_interest"]
        trade_date        = state.get("trade_date", "N/A")
        horizon           = state.get("trading_horizon", "short")
        risk_debate_state = state["risk_debate_state"]
        history           = risk_debate_state["history"]

        trader_plan            = state["investment_plan"]

        
        if horizon == "short":
            horizon_note = (
                "Chiến lược đang đánh giá là LƯỚT SÓNG NGẮN HẠN (2-5 ngày). "
                "Tập trung đánh giá rủi ro trong khung thời gian này. "
                "Stop-loss và điểm thoát là ưu tiên hàng đầu.\n"
                "Ba quyết định hợp lệ: BUY, SELL, HOLD."
            )
            decision_section = (
                "#### Quyết Định: **[BUY / SELL / HOLD]**\n\n"

                "#### Thông Số Thực Thi\n"
                "- Vùng giá vào lệnh: [nếu BUY/SELL]\n"
                "- Stop-loss: [mức giá hoặc %]\n"
                "- Mục tiêu chốt lời: [mức giá hoặc %]\n"
                "- Khung nắm giữ: [X–Y ngày]\n\n"

                "#### Điều Kiện Xem Xét Lại\n"
                "- Nên đổi sang BUY nếu: [điều kiện cụ thể]\n"
                "- Nên đổi sang SELL nếu: [điều kiện cụ thể]\n\n"

                "#### Quản Lý Rủi Ro Ngắn Hạn\n"
                "- Rủi ro chính cần theo dõi: [1-2 rủi ro quan trọng nhất]"
            )
        else:
            horizon_note = (
                "Chiến lược đang đánh giá là ĐẦU TƯ DÀI HẠN. "
                "Chỉ có hai quyết định hợp lệ: BUY hoặc NOT BUY. "
                "Không đánh giá biến động ngắn hạn — chỉ tập trung vào rủi ro có thể phá vỡ thesis dài hạn.\n"
                "Nếu không đủ chắc chắn, kết quả mặc định là NOT BUY."
            )
            decision_section = (
                "#### Quyết Định: **[BUY / NOT BUY]**\n\n"

                "#### Điều Kiện Theo Dõi\n"
                "- Nếu BUY: mốc xem xét thoát (định giá đạt mục tiêu, thesis thay đổi)\n"
                "- Nếu NOT BUY: điều kiện nào sẽ khiến xem xét lại\n\n"

                "#### Rủi Ro Dài Hạn\n"
                "- Rủi ro có thể phá vỡ thesis: [1-2 rủi ro cấu trúc quan trọng nhất]"
            )

        prompt = (
            f"Bạn là Portfolio Manager — người ra quyết định giao dịch cuối cùng cho {company_name}.\n\n"

            f"## CHIẾN LƯỢC HIỆN TẠI\n{horizon_note}\n\n"

            "## TIÊU CHUẨN ĐÁNH GIÁ\n"
            "Đánh giá bằng chứng từ ba analyst:\n"
            "- **Risky Analyst**: Phát hiện upside, cơ hội bị bỏ lỡ do quá thận trọng\n"
            "- **Safe Analyst**: Phát hiện downside, rủi ro bị đánh giá thấp\n"
            "- **Neutral Analyst**: Cân bằng hai bên, chỉ ra điểm yếu của cả hai\n\n"
            "Mỗi analyst có xu hướng cố hữu — hãy đánh giá CHẤT LƯỢNG BẰNG CHỨNG, "
            "không phải chiều hướng của analyst.\n\n"

            "## QUY TRÌNH RA QUYẾT ĐỊNH\n"
            "Bước 1: Xác định luận điểm có bằng chứng cụ thể nhất từ mỗi bên\n"
            "Bước 2: Đối chiếu với tín hiệu Alpha (nếu có) và kế hoạch của Research team\n"
            "Bước 3: Chọn quyết định có bằng chứng thuyết phục nhất\n\n"

            f"**Kế hoạch từ Research team:**\n{sanitize_for_prompt(trader_plan)}\n\n"

            f"**Toàn bộ lịch sử tranh luận:**\n{sanitize_for_prompt(history)}\n\n"

            "---\n"
            f"## YÊU CẦU OUTPUT (BẮT BUỘC tuân theo format)\n\n"

            f"### Quyết Định Cuối Cùng — {company_name} — {trade_date}\n\n"

            "#### Đánh Giá Tranh Luận\n"
            "**Luận điểm mạnh nhất ủng hộ tăng giá / mua:** [từ Risky hoặc bất kỳ analyst]\n"
            "**Luận điểm mạnh nhất ủng hộ giảm / không mua:** [từ Safe hoặc bất kỳ analyst]\n"
            "**Tín hiệu Alpha:** [hướng và mức độ tin cậy]\n\n"

            "#### Lý Do Quyết Định\n"
            "[2-3 câu giải thích tại sao bằng chứng dẫn đến quyết định này]\n\n"

            f"{decision_section}"
        )

        response = llm.invoke(prompt)

        new_risk_debate_state = {
            "judge_decision":            response.content,
            "history":                   risk_debate_state["history"],
            "risky_history":             risk_debate_state["risky_history"],
            "safe_history":              risk_debate_state["safe_history"],
            "neutral_history":           risk_debate_state["neutral_history"],
            "latest_speaker":            "Judge",
            "current_risky_response":    risk_debate_state["current_risky_response"],
            "current_safe_response":     risk_debate_state["current_safe_response"],
            "current_neutral_response":  risk_debate_state["current_neutral_response"],
            "count":                     risk_debate_state["count"],
        }

        return {
            "risk_debate_state": new_risk_debate_state,
            "final_trade_decision": response.content,
        }

    return risk_manager_node
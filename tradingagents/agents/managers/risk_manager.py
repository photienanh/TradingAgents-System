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
        research_plan     = state["investment_plan"]
        trader_plan       = state["trader_investment_plan"]

        
        if horizon == "short":
            horizon_note = (
                "Đây là quyết định cho chiến lược NGẮN HẠN (2-5 ngày). "
                "Ba lựa chọn hợp lệ: BUY, SELL, HOLD."
            )
            decision_section = (
                "#### Quyết Định: **[BUY / SELL / HOLD]**\n\n"
                "#### Thông số thực thi\n"
                "[Đưa ra thông số thực thi phù hợp nhất]"
                "#### Lý Do\n"
                "[Lý do dẫn đến quyết định, dựa trên bằng chứng từ tranh luận]"
            )
        else:
            horizon_note = (
                "Đây là quyết định cho chiến lược ĐẦU TƯ DÀI HẠN. "
                "Hai lựa chọn hợp lệ: BUY hoặc NOT BUY."
            )
            decision_section = (
                "#### Quyết Định: **[BUY / NOT BUY]**\n\n"
                "#### Thông số thực thi\n"
                "[Đưa ra thông số thực thi phù hợp nhất nếu có, hoặc ghi N/A nếu không có]"
                "#### Lý Do\n"
                "[Lý do dẫn đến quyết định, dựa trên bằng chứng từ tranh luận]"
            )

        prompt = (
            f"Bạn là Portfolio Manager - người ra quyết định giao dịch cuối cùng cho {company_name}.\n\n"
            "Trước đó Research Manager đã đề xuất định hướng và Trader đã đề xuất thông số thực thi giao dịch. "
            "Nhóm Quản trị Rủi ro (Risky Team) vừa tiến hành đánh giá cả định hướng lẫn thông số này.\n\n"

            f"## BỐI CẢNH\n{horizon_note}\n\n"

            f"## KẾT QUẢ NGHIÊN CỨU VÀ QUYẾT ĐỊNH CỦA RESEARCH TEAM\n{sanitize_for_prompt(research_plan)}\n\n"

            f"## KẾ HOẠCH GIAO DỊCH TỪ TRADER\n{sanitize_for_prompt(trader_plan)}\n\n"

            f"## LỊCH SỬ TRANH LUẬN TỪ RISKY TEAM\n{sanitize_for_prompt(history)}\n\n"

            "- **Risky Analyst**: Góc nhìn upside\n"
            "- **Safe Analyst**: Góc nhìn downside\n"
            "- **Neutral Analyst**: Cân bằng hai bên, chỉ ra điểm yếu, điểm mạnh của cả hai\n\n"

            "## QUY TRÌNH RA QUYẾT ĐỊNH\n"
            "Bước 1: Xác định luận điểm có bằng chứng cụ thể nhất từ mỗi bên\n"
            "Bước 2: Chọn quyết định có bằng chứng thuyết phục nhất\n"
            "Bước 3: Đưa ra quyết định cuối cùng\n\n"

            f"## YÊU CẦU OUTPUT (BẮT BUỘC tuân theo format)\n\n"
            f"### Quyết Định Cuối Cùng - {company_name} - {trade_date}\n\n"

            "#### Đánh Giá Tranh Luận\n"
            "**Luận điểm mạnh nhất ủng hộ upside:** [từ Risky hoặc bất kỳ analyst nào]\n"
            "**Luận điểm mạnh nhất ủng hộ downside:** [từ Safe hoặc bất kỳ analyst nào]\n"

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
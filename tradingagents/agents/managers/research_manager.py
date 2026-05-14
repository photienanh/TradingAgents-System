from tradingagents.agents.utils.text_sanitize import sanitize_for_prompt


def create_research_manager(llm):
    def research_manager_node(state) -> dict:
        horizon    = state.get("trading_horizon", "short")
        ticker     = state.get("company_of_interest", "N/A")
        trade_date = state.get("trade_date", "N/A")

        history                 = state["investment_debate_state"].get("history", "")
        investment_debate_state = state["investment_debate_state"]


        if horizon == "short":
            horizon_note = (
                "Đây là quyết định cho chiến lược NGẮN HẠN (2-5 ngày). "
                "Ba quyết định hợp lệ: BUY, SELL, HOLD."
            )
        else:
            horizon_note = (
                "Đây là quyết định cho chiến lược ĐẦU TƯ DÀI HẠN. "
                "Hai quyết định hợp lệ: BUY hoặc NOT BUY."
            )

        decision_choices = "BUY/SELL/HOLD" if horizon == "short" else "BUY hoặc NOT BUY"

        prompt = (
            f"Bạn là Research Manager - người tổng hợp cuộc tranh luận giữa Bull và Bear để đưa ra quyết định đầu tư cho {ticker}.\n\n"

            f"## BỐI CẢNH\n{horizon_note}\n\n"

            "## NGUYÊN TẮC (BẮT BUỘC)\n"
            "Chỉ chất lượng bằng chứng quyết định kết quả - không có chiều hướng nào là 'mặc định' hay 'an toàn hơn'.\n"

            f"## LỊCH SỬ TRANH LUẬN\n{sanitize_for_prompt(history)}\n\n"

            "---\n"
            f"## YÊU CẦU OUTPUT\n\n"
            f"### Quyết Định Nhóm Nghiên Cứu - {ticker} - {trade_date}\n\n"

            "#### Tóm Tắt\n"
            "**Bull:** [Luận điểm có bằng chứng thuyết phục nhất]\n"
            "**Bear:** [Luận điểm có bằng chứng thuyết phục nhất]\n\n"

            "#### Đánh Giá\n"
            "[2-3 câu: bên nào có bằng chứng thuyết phục hơn và tại sao]\n\n"

            f"#### Quyết Định: **{decision_choices}**\n\n"

            "#### Lý Do\n"
            "[Lý do dẫn đến quyết định trên, dựa trên bằng chứng từ tranh luận]"
        )

        response = llm.invoke(prompt)

        new_investment_debate_state = {
            "judge_decision":  response.content,
            "history":         investment_debate_state.get("history", ""),
            "bear_history":    investment_debate_state.get("bear_history", ""),
            "bull_history":    investment_debate_state.get("bull_history", ""),
            "current_response": response.content,
            "count":           investment_debate_state["count"],
        }

        return {
            "investment_debate_state": new_investment_debate_state,
            "investment_plan":         response.content,
        }

    return research_manager_node
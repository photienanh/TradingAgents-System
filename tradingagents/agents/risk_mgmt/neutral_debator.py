"""
tradingagents/agents/risk_mgmt/neutral_debator.py
"""
from tradingagents.agents.utils.text_sanitize import sanitize_for_prompt


def create_neutral_debator(llm):
    def neutral_node(state) -> dict:
        horizon = state.get("trading_horizon", "short")
        risk_debate_state = state["risk_debate_state"]
        history         = risk_debate_state.get("history", "")
        neutral_history = risk_debate_state.get("neutral_history", "")

        current_risky_response    = risk_debate_state.get("current_risky_response", "")
        current_safe_response     = risk_debate_state.get("current_safe_response", "")

        research_decision      = state["investment_plan"]
        trader_decision        = state["trader_investment_plan"]

        if horizon == "short":
            horizon_note = "Đây là tranh luận về chiến lược NGẮN HẠN (2-5 ngày)."
        else:
            horizon_note = "Đây là tranh luận về chiến lược ĐẦU TƯ DÀI HẠN."

        prompt = (
            f"{horizon_note}\n\n"
            "Bạn là Neutral Analyst trong nhóm quản lý rủi ro. Vai trò của bạn là đánh giá khách quan, "
            "không có lập trường mặc định nào. Bạn chỉ ra điểm yếu của cả hai bên và tổng hợp bức tranh toàn cảnh.\n\n"
            "Nhiệm vụ của bạn là làm trọng tài, đánh giá khách quan về TỶ LỆ RISK/REWARD TỔNG THỂ, bao gồm cả:\n"
            "- Tính đúng đắn trong định hướng của Research Manager.\n"
            "- Tính hợp lý trong thông số của Trader.\n\n"
            
            f"## Kế Hoạch Định Hướng:\n{sanitize_for_prompt(research_decision)}\n\n"
            f"## Kế Hoạch Thực Thi:\n{sanitize_for_prompt(trader_decision)}\n\n"
            f"## Lịch sử tranh luận\n{sanitize_for_prompt(history)}\n\n"
            f"Risky Analyst: {sanitize_for_prompt(current_risky_response)}\n"
            f"Safe Analyst: {sanitize_for_prompt(current_safe_response)}\n\n"
            
            "Hãy tổng hợp quan điểm khách quan. Chỉ ra điểm yếu và điểm mạnh cụ thể của mỗi bên. "
            "Đưa ra bức tranh toàn cảnh trung thực nhất. "
            "Viết theo phong cách hội thoại tự nhiên."
        )

        response = llm.invoke(prompt)
        argument = f"Neutral Analyst: {response.content}"

        new_risk_debate_state = {
            "history":                  history + "\n" + argument,
            "risky_history":            risk_debate_state.get("risky_history", ""),
            "safe_history":             risk_debate_state.get("safe_history", ""),
            "neutral_history":          neutral_history + "\n" + argument,
            "latest_speaker":           "Neutral",
            "current_risky_response":   risk_debate_state.get("current_risky_response", ""),
            "current_safe_response":    risk_debate_state.get("current_safe_response", ""),
            "current_neutral_response": argument,
            "count":                    risk_debate_state["count"] + 1,
        }
        return {"risk_debate_state": new_risk_debate_state}

    return neutral_node
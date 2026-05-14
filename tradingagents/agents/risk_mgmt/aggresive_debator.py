"""
tradingagents/agents/risk_mgmt/aggresive_debator.py
"""
from tradingagents.agents.utils.text_sanitize import sanitize_for_prompt


def create_risky_debator(llm):
    def risky_node(state) -> dict:
        horizon = state.get("trading_horizon", "short")
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")
        risky_history = risk_debate_state.get("risky_history", "")

        current_safe_response    = risk_debate_state.get("current_safe_response", "")
        current_neutral_response = risk_debate_state.get("current_neutral_response", "")

        market_research_report = state["market_report"]
        sentiment_report       = state["sentiment_report"]
        news_report            = state["news_report"]
        fundamentals_report    = state["fundamentals_report"]
        quant_report           = state["quant_report"]
        trader_decision        = state["trader_investment_plan"]

        if horizon == "short":
            horizon_note = "Đây là tranh luận về chiến lược NGẮN HẠN (2–5 ngày)."
            alpha_context = (
                f"\nTín hiệu định lượng (Alpha): {sanitize_for_prompt(quant_report)}\n"
            ) if quant_report else ""
        else:
            horizon_note = "Đây là tranh luận về chiến lược ĐẦU TƯ DÀI HẠN."
            alpha_context = ""

        prompt = (
            f"{horizon_note}\n\n"
            "Bạn là Risky Analyst trong nhóm quản lý rủi ro. Vai trò của bạn là đại diện cho góc nhìn upside: "
            "nhận diện và bảo vệ các cơ hội lợi nhuận, phản biện những lập luận quá thận trọng dẫn đến bỏ lỡ cơ hội.\n\n"
            "Đọc toàn bộ dữ liệu và tự xác định bằng chứng nào ủng hộ mạnh nhất cho luận điểm của bạn.\n\n"
            f"Kế hoạch của Trader:\n{sanitize_for_prompt(trader_decision)}\n"
            f"{alpha_context}"
            f"Phân tích thị trường: {sanitize_for_prompt(market_research_report)}\n"
            f"Tâm lý & mạng xã hội: {sanitize_for_prompt(sentiment_report)}\n"
            f"Tin tức: {sanitize_for_prompt(news_report)}\n"
            f"Tài chính doanh nghiệp: {sanitize_for_prompt(fundamentals_report)}\n\n"
            f"## Lịch sử tranh luận\n{sanitize_for_prompt(history)}\n\n"
            f"Safe Analyst: {sanitize_for_prompt(current_safe_response)}\n"
            f"Neutral Analyst: {sanitize_for_prompt(current_neutral_response)}\n\n"
            "Phản biện trực tiếp các lo ngại của Safe Analyst bằng bằng chứng cụ thể. "
            "Chỉ ra nơi họ đang phóng đại rủi ro hoặc bỏ qua cơ hội. "
            "Viết theo phong cách hội thoại tranh luận tự nhiên."
        )

        response = llm.invoke(prompt)
        argument = f"Risky Analyst: {response.content}"

        new_risk_debate_state = {
            "history":                  history + "\n" + argument,
            "risky_history":            risky_history + "\n" + argument,
            "safe_history":             risk_debate_state.get("safe_history", ""),
            "neutral_history":          risk_debate_state.get("neutral_history", ""),
            "latest_speaker":           "Risky",
            "current_risky_response":   argument,
            "current_safe_response":    risk_debate_state.get("current_safe_response", ""),
            "current_neutral_response": risk_debate_state.get("current_neutral_response", ""),
            "count":                    risk_debate_state["count"] + 1,
        }
        return {"risk_debate_state": new_risk_debate_state}

    return risky_node
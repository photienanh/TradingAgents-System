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
            "Bạn là Neutral Analyst trong nhóm quản lý rủi ro. Vai trò của bạn là đánh giá khách quan, "
            "không có lập trường mặc định nào. Bạn chỉ ra điểm yếu của cả hai bên và tổng hợp bức tranh toàn cảnh.\n\n"
            "Đọc toàn bộ dữ liệu, đánh giá luận điểm nào có bằng chứng thực sự và luận điểm nào chỉ là suy diễn. "
            "Neutral không có nghĩa là luôn chọn HOLD - nếu bằng chứng nghiêng rõ về một phía, hãy nói thẳng.\n\n"
            f"Kế hoạch của Trader:\n{sanitize_for_prompt(trader_decision)}\n"
            f"{alpha_context}\n\n"
            f"Phân tích thị trường: {sanitize_for_prompt(market_research_report)}\n"
            f"Tâm lý & mạng xã hội: {sanitize_for_prompt(sentiment_report)}\n"
            f"Tin tức: {sanitize_for_prompt(news_report)}\n"
            f"Tài chính doanh nghiệp: {sanitize_for_prompt(fundamentals_report)}\n\n"
            f"## Lịch sử tranh luận\n{sanitize_for_prompt(history)}\n\n"
            f"Risky Analyst: {sanitize_for_prompt(current_risky_response)}\n"
            f"Safe Analyst: {sanitize_for_prompt(current_safe_response)}\n\n"
            "Đánh giá khách quan cả hai bên. Chỉ ra điểm yếu và điểm mạnh cụ thể của mỗi bên. "
            "Tổng hợp bức tranh risk/reward một cách trung thực. "
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
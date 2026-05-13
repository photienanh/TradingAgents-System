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

        alpha_context = ""
        if quant_report:
            alpha_context = f"\nTín hiệu định lượng Alpha: {sanitize_for_prompt(quant_report)}"
        
        horizon_note = (
            "Chiến lược đang đánh giá là LƯỚT SÓNG NGẮN HẠN (2-5 ngày). "
            "Tập trung vào rủi ro và cơ hội trong khung thời gian này."
        ) if horizon == "short" else (
            "Chiến lược đang đánh giá là ĐẦU TƯ DÀI HẠN. "
            "Quyết định cuối cùng chỉ là BUY hoặc NOT BUY. "
            "Không đánh giá biến động ngắn hạn — chỉ rủi ro có thể phá vỡ luận điểm dài hạn."
        )

        prompt = (
            f"{horizon_note}\n\n"
            "Bạn là Risky Analyst trong nhóm quản lý rủi ro. Vai trò của bạn là đại diện cho góc nhìn 'upside-focused': "
            "nhận diện và bảo vệ các cơ hội lợi nhuận, chống lại xu hướng quá thận trọng dẫn đến bỏ lỡ cơ hội.\n\n"

            "**Trọng tâm phân tích của bạn:**\n"
            "- Các catalyst tăng giá đang bị đánh giá thấp\n"
            "- Upside potential cụ thể dựa trên dữ liệu\n"
            "- Điểm yếu trong lập luận quá thận trọng của Safe Analyst\n"
            "- Chi phí cơ hội của việc không hành động\n"
            "- Nếu tín hiệu Alpha ủng hộ long, đây là bằng chứng định lượng bổ sung\n\n"

            "**Lưu ý quan trọng**: Vai trò của bạn không phải lúc nào cũng là ủng hộ BUY. "
            "Nếu kế hoạch của Trader là SELL, bạn bảo vệ lý do SELL đó bằng lập luận upside của bear case.\n\n"

            f"Kế hoạch của Trader:\n{sanitize_for_prompt(trader_decision)}\n"
            f"{sanitize_for_prompt(alpha_context)}\n\n"

            f"Dữ liệu thị trường: {sanitize_for_prompt(market_research_report)}\n"
            f"Tâm lý: {sanitize_for_prompt(sentiment_report)}\n"
            f"Tin tức: {sanitize_for_prompt(news_report)}\n"
            f"Tài chính doanh nghiệp: {sanitize_for_prompt(fundamentals_report)}\n"
            f"Lịch sử tranh luận: {sanitize_for_prompt(history)}\n"
            f"Safe Analyst: {sanitize_for_prompt(current_safe_response)}\n"
            f"Neutral Analyst: {sanitize_for_prompt(current_neutral_response)}\n\n"

            "Phản biện trực tiếp các lo ngại của Safe Analyst bằng dữ liệu cụ thể. "
            "Chỉ ra nơi họ đang phóng đại rủi ro hoặc bỏ qua cơ hội. "
            "Viết theo phong cách hội thoại tự nhiên."
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
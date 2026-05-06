"""
tradingagents/agents/risk_mgmt/conservative_debator.py
"""
from tradingagents.agents.utils.text_sanitize import sanitize_for_prompt


def create_safe_debator(llm):
    def safe_node(state) -> dict:
        horizon = state.get("trading_horizon", "short")
        risk_debate_state = state["risk_debate_state"]
        history      = risk_debate_state.get("history", "")
        safe_history = risk_debate_state.get("safe_history", "")

        current_risky_response    = risk_debate_state.get("current_risky_response", "")
        current_neutral_response  = risk_debate_state.get("current_neutral_response", "")
        current_alphagpt_response = risk_debate_state.get("current_alphagpt_response", "")

        market_research_report = state["market_report"]
        sentiment_report       = state["sentiment_report"]
        news_report            = state["news_report"]
        fundamentals_report    = state["fundamentals_report"]
        trader_decision        = state["trader_investment_plan"]

        alphagpt_context = ""
        if current_alphagpt_response:
            alphagpt_context = f"\nTín hiệu định lượng AlphaGPT: {sanitize_for_prompt(current_alphagpt_response)}"

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
            "Bạn là Safe/Conservative Analyst trong nhóm quản lý rủi ro. Vai trò của bạn là đại diện cho góc nhìn 'downside-focused': "
            "nhận diện các rủi ro bị đánh giá thấp, bảo vệ danh mục khỏi thua lỗ không cần thiết.\n\n"

            "**Trọng tâm phân tích của bạn:**\n"
            "- Các rủi ro cụ thể và có bằng chứng (không phải lo ngại chung chung)\n"
            "- Downside scenario: nếu điều gì đó đi sai, mức độ tổn thất là bao nhiêu?\n"
            "- Giả định lạc quan nào trong kế hoạch của Trader có thể không đúng?\n"
            "- Điểm yếu trong lập luận của Risky Analyst\n"
            "- Nếu tín hiệu AlphaGPT cho thấy tín hiệu yếu hoặc short, đây là bằng chứng định lượng hỗ trợ thận trọng\n\n"

            "**Lưu ý quan trọng**: Nhiệm vụ của bạn là kiểm tra tính vững chắc của kế hoạch — "
            "không phải tự động phản đối BUY hay ủng hộ HOLD. "
            "Nếu kế hoạch Trader là SELL, bạn kiểm tra xem SELL có thực sự justified không.\n\n"

            f"Kế hoạch của Trader:\n{sanitize_for_prompt(trader_decision)}\n"
            f"{sanitize_for_prompt(alphagpt_context)}\n\n"

            f"Dữ liệu thị trường: {sanitize_for_prompt(market_research_report)}\n"
            f"Tâm lý: {sanitize_for_prompt(sentiment_report)}\n"
            f"Tin tức: {sanitize_for_prompt(news_report)}\n"
            f"Tài chính doanh nghiệp: {sanitize_for_prompt(fundamentals_report)}\n"
            f"Lịch sử tranh luận: {sanitize_for_prompt(history)}\n"
            f"Risky Analyst: {sanitize_for_prompt(current_risky_response)}\n"
            f"Neutral Analyst: {sanitize_for_prompt(current_neutral_response)}\n\n"

            "Phản biện trực tiếp lập luận của Risky Analyst. "
            "Chỉ ra cụ thể đâu là rủi ro thực sự (không phải rủi ro lý thuyết). "
            "Thừa nhận điểm mạnh của Risky nếu có. "
            "Viết theo phong cách hội thoại tự nhiên."
        )

        response = llm.invoke(prompt)
        argument = f"Safe Analyst: {response.content}"

        new_risk_debate_state = {
            "history":                  history + "\n" + argument,
            "risky_history":            risk_debate_state.get("risky_history", ""),
            "safe_history":             safe_history + "\n" + argument,
            "neutral_history":          risk_debate_state.get("neutral_history", ""),
            "alphagpt_history":         risk_debate_state.get("alphagpt_history", ""),
            "latest_speaker":           "Safe",
            "current_risky_response":   risk_debate_state.get("current_risky_response", ""),
            "current_safe_response":    argument,
            "current_neutral_response": risk_debate_state.get("current_neutral_response", ""),
            "current_alphagpt_response": current_alphagpt_response,
            "count":                    risk_debate_state["count"] + 1,
        }
        return {"risk_debate_state": new_risk_debate_state}

    return safe_node
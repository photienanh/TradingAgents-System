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
            "Bạn là Safe Analyst trong nhóm quản lý rủi ro. Vai trò của bạn là đại diện cho góc nhìn downside: "
            "nhận diện các rủi ro bị đánh giá thấp, kiểm tra tính vững chắc của kế hoạch, bảo vệ danh mục khỏi thua lỗ không cần thiết.\n\n"
            "Đọc toàn bộ dữ liệu và tự xác định bằng chứng nào cho thấy kế hoạch đang chứa rủi ro thực sự. "
            "Nhiệm vụ của bạn là kiểm tra, không phải tự động phản đối BUY hay ủng hộ HOLD. "
            "Nếu kế hoạch Trader là SELL, hãy kiểm tra xem SELL có thực sự hợp lý hay không.\n\n"
            f"Kế hoạch của Trader:\n{sanitize_for_prompt(trader_decision)}\n"
            f"{alpha_context}"
            f"Phân tích thị trường: {sanitize_for_prompt(market_research_report)}\n"
            f"Tâm lý & mạng xã hội: {sanitize_for_prompt(sentiment_report)}\n"
            f"Tin tức: {sanitize_for_prompt(news_report)}\n"
            f"Tài chính doanh nghiệp: {sanitize_for_prompt(fundamentals_report)}\n\n"
            f"## Lịch sử tranh luận\n{sanitize_for_prompt(history)}\n\n"
            f"Risky Analyst: {sanitize_for_prompt(current_risky_response)}\n"
            f"Neutral Analyst: {sanitize_for_prompt(current_neutral_response)}\n\n"
            "Phản biện trực tiếp lập luận của Risky Analyst. "
            "Chỉ ra đâu là rủi ro thực sự có bằng chứng, không phải rủi ro lý thuyết. "
            "Thừa nhận điểm mạnh của Risky khi có. "
            "Viết theo phong cách hội thoại tranh luận tự nhiên."
        )

        response = llm.invoke(prompt)
        argument = f"Safe Analyst: {response.content}"

        new_risk_debate_state = {
            "history":                  history + "\n" + argument,
            "risky_history":            risk_debate_state.get("risky_history", ""),
            "safe_history":             safe_history + "\n" + argument,
            "neutral_history":          risk_debate_state.get("neutral_history", ""),
            "latest_speaker":           "Safe",
            "current_risky_response":   risk_debate_state.get("current_risky_response", ""),
            "current_safe_response":    argument,
            "current_neutral_response": risk_debate_state.get("current_neutral_response", ""),
            "count":                    risk_debate_state["count"] + 1,
        }
        return {"risk_debate_state": new_risk_debate_state}

    return safe_node
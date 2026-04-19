"""
tradingagents/agents/risk_mgmt/neutral_debator.py
"""
import json

from tradingagents.agents.utils.text_sanitize import sanitize_for_prompt


def create_neutral_debator(llm):
    def neutral_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        history         = risk_debate_state.get("history", "")
        neutral_history = risk_debate_state.get("neutral_history", "")

        current_risky_response    = risk_debate_state.get("current_risky_response", "")
        current_safe_response     = risk_debate_state.get("current_safe_response", "")
        current_alphagpt_response = risk_debate_state.get("current_alphagpt_response", "")

        market_research_report = state["market_report"]
        sentiment_report       = state["sentiment_report"]
        news_report            = state["news_report"]
        fundamentals_report    = state["fundamentals_report"]
        trader_decision        = state["trader_investment_plan"]

        alphagpt_context = ""
        if current_alphagpt_response:
            alphagpt_context = f"\nTín hiệu định lượng AlphaGPT: {sanitize_for_prompt(current_alphagpt_response)}"

        prompt = (
            "Bạn là Neutral Analyst trong nhóm quản lý rủi ro. Vai trò của bạn là đánh giá khách quan — "
            "không có lập trường mặc định nào. Bạn chỉ ra điểm yếu của CẢ HAI bên và tổng hợp bức tranh toàn cảnh.\n\n"

            "**Trọng tâm phân tích của bạn:**\n"
            "- Luận điểm nào của Risky và Safe có bằng chứng thực sự, luận điểm nào chỉ là suy đoán?\n"
            "- Bức tranh tổng thể từ góc nhìn risk/reward: upside và downside tương quan như thế nào?\n"
            "- Tín hiệu AlphaGPT nói gì và mức độ tin cậy của nó?\n"
            "- Điều kiện nào sẽ khiến kịch bản bull hoặc bear đúng?\n"
            "- Có chiến lược nào phù hợp hơn BUY/SELL/HOLD thuần túy không (ví dụ: position sizing, staged entry)?\n\n"

            "**Quan trọng**: Neutral không có nghĩa là luôn chọn HOLD. "
            "Nếu bằng chứng rõ ràng nghiêng về một phía, hãy nói thẳng.\n\n"

            f"Kế hoạch của Trader:\n{sanitize_for_prompt(trader_decision)}\n"
            f"{sanitize_for_prompt(alphagpt_context)}\n\n"

            f"Dữ liệu thị trường: {sanitize_for_prompt(market_research_report)}\n"
            f"Tâm lý: {sanitize_for_prompt(sentiment_report)}\n"
            f"Tin tức: {sanitize_for_prompt(news_report)}\n"
            f"Cơ bản: {sanitize_for_prompt(fundamentals_report)}\n"
            f"Lịch sử tranh luận: {sanitize_for_prompt(history)}\n"
            f"Risky Analyst: {sanitize_for_prompt(current_risky_response)}\n"
            f"Safe Analyst: {sanitize_for_prompt(current_safe_response)}\n\n"

            "Đánh giá khách quan cả hai bên. Chỉ ra điểm yếu cụ thể của mỗi bên. "
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
            "alphagpt_history":         risk_debate_state.get("alphagpt_history", ""),
            "latest_speaker":           "Neutral",
            "current_risky_response":   risk_debate_state.get("current_risky_response", ""),
            "current_safe_response":    risk_debate_state.get("current_safe_response", ""),
            "current_neutral_response": argument,
            "current_alphagpt_response": current_alphagpt_response,
            "count":                    risk_debate_state["count"] + 1,
        }
        return {"risk_debate_state": new_risk_debate_state}

    return neutral_node
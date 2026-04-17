"""
tradingagents/agents/risk_mgmt/neutral_debator.py

FIX: Thêm _s() sanitizer để tránh lỗi 400 JSON malformed khi gửi lên OpenAI API.
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
            alphagpt_context = f"\nTín hiệu định lượng AlphaGPT (tham khảo): {sanitize_for_prompt(current_alphagpt_response)}"

        prompt = (
            "Bạn là Neutral Risk Analyst. Vai trò của bạn là cung cấp góc nhìn cân bằng, cân nhắc cả "
            "lợi ích tiềm năng lẫn rủi ro của quyết định/kế hoạch từ trader. Bạn ưu tiên cách tiếp "
            "cận toàn diện: đánh giá cả mặt tích cực và tiêu cực, đồng thời xét thêm xu hướng thị "
            "trường rộng hơn, dịch chuyển kinh tế có thể xảy ra và chiến lược đa dạng hóa. Đây là "
            "quyết định của trader:\n\n"
            f"    {sanitize_for_prompt(trader_decision)}\n"
            f"    {sanitize_for_prompt(alphagpt_context)}\n\n"
            "Nhiệm vụ của bạn là phản biện cả Risky Analyst và Safe Analyst, chỉ ra nơi mỗi bên có "
            "thể quá lạc quan hoặc quá thận trọng. Nếu có tín hiệu AlphaGPT, hãy dùng như bằng chứng "
            "trung lập để kiểm chứng cả hai hướng tranh luận. Hãy dùng dữ liệu sau:\n\n"
            f"Báo cáo nghiên cứu thị trường: {sanitize_for_prompt(market_research_report)}\n"
            f"Báo cáo tâm lý mạng xã hội: {sanitize_for_prompt(sentiment_report)}\n"
            f"Báo cáo thời sự thế giới gần đây: {sanitize_for_prompt(news_report)}\n"
            f"Báo cáo cơ bản doanh nghiệp: {sanitize_for_prompt(fundamentals_report)}\n"
            f"Lịch sử hội thoại hiện tại: {sanitize_for_prompt(history)}\n"
            f"Phản hồi gần nhất của Risky Analyst: {sanitize_for_prompt(current_risky_response)}\n"
            f"Phản hồi gần nhất của Safe Analyst: {sanitize_for_prompt(current_safe_response)}\n"
            "Nếu các góc nhìn còn lại chưa có phản hồi, không bịa nội dung; chỉ trình bày lập luận "
            "của bạn.\n\n"
            "Hãy chủ động phân tích phê phán cả hai phía, xử lý điểm yếu trong lập luận rủi ro cao "
            "lẫn lập luận quá bảo thủ để bảo vệ một cách tiếp cận cân bằng hơn. Chất vấn từng luận "
            "điểm của họ để cho thấy vì sao chiến lược rủi ro trung dung có thể dung hòa tăng trưởng "
            "và kiểm soát biến động cực đoan. Trình bày theo phong cách hội thoại tự nhiên, không cần "
            "định dạng đặc biệt."
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
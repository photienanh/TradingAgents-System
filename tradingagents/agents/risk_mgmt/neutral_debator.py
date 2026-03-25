import time
import json


def create_neutral_debator(llm):
    def neutral_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")
        neutral_history = risk_debate_state.get("neutral_history", "")

        current_risky_response = risk_debate_state.get("current_risky_response", "")
        current_safe_response = risk_debate_state.get("current_safe_response", "")

        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        trader_decision = state["trader_investment_plan"]

        prompt = f"""Bạn là Neutral Risk Analyst. Vai trò của bạn là cung cấp góc nhìn cân bằng, cân nhắc cả lợi ích tiềm năng lẫn rủi ro của quyết định/kế hoạch từ trader. Bạn ưu tiên cách tiếp cận toàn diện: đánh giá cả mặt tích cực và tiêu cực, đồng thời xét thêm xu hướng thị trường rộng hơn, dịch chuyển kinh tế có thể xảy ra và chiến lược đa dạng hóa. Đây là quyết định của trader:

    {trader_decision}

    Nhiệm vụ của bạn là phản biện cả Risky Analyst và Safe Analyst, chỉ ra nơi mỗi bên có thể quá lạc quan hoặc quá thận trọng. Hãy dùng insight từ các nguồn dữ liệu sau để đề xuất hướng điều chỉnh quyết định theo chiến lược rủi ro vừa phải và bền vững:

    Báo cáo nghiên cứu thị trường: {market_research_report}
    Báo cáo tâm lý mạng xã hội: {sentiment_report}
    Báo cáo thời sự thế giới gần đây: {news_report}
    Báo cáo cơ bản doanh nghiệp: {fundamentals_report}
    Lịch sử hội thoại hiện tại: {history}
    Phản hồi gần nhất của Risky Analyst: {current_risky_response}
    Phản hồi gần nhất của Safe Analyst: {current_safe_response}
    Nếu các góc nhìn còn lại chưa có phản hồi, không bịa nội dung; chỉ trình bày lập luận của bạn.

    Hãy chủ động phân tích phê phán cả hai phía, xử lý điểm yếu trong lập luận rủi ro cao lẫn lập luận quá bảo thủ để bảo vệ một cách tiếp cận cân bằng hơn. Chất vấn từng luận điểm của họ để cho thấy vì sao chiến lược rủi ro trung dung có thể dung hòa tăng trưởng và kiểm soát biến động cực đoan. Tập trung vào tranh luận thay vì chỉ trình bày dữ liệu, nhằm chứng minh rằng góc nhìn cân bằng thường dẫn đến kết quả ổn định hơn. Trình bày theo phong cách hội thoại tự nhiên, không cần định dạng đặc biệt."""

        response = llm.invoke(prompt)

        argument = f"Neutral Analyst: {response.content}"

        new_risk_debate_state = {
            "history": history + "\n" + argument,
            "risky_history": risk_debate_state.get("risky_history", ""),
            "safe_history": risk_debate_state.get("safe_history", ""),
            "neutral_history": neutral_history + "\n" + argument,
            "alphagpt_history": risk_debate_state.get("alphagpt_history", ""),
            "latest_speaker": "Neutral",
            "current_risky_response": risk_debate_state.get(
                "current_risky_response", ""
            ),
            "current_safe_response": risk_debate_state.get("current_safe_response", ""),
            "current_neutral_response": argument,
            "current_alphagpt_response": risk_debate_state.get("current_alphagpt_response", ""),
            "count": risk_debate_state["count"] + 1,
        }

        return {"risk_debate_state": new_risk_debate_state}

    return neutral_node

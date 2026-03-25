from langchain_core.messages import AIMessage
import time
import json


def create_safe_debator(llm):
    def safe_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")
        safe_history = risk_debate_state.get("safe_history", "")

        current_risky_response = risk_debate_state.get("current_risky_response", "")
        current_neutral_response = risk_debate_state.get("current_neutral_response", "")

        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        trader_decision = state["trader_investment_plan"]

        prompt = f"""Bạn là Safe/Conservative Risk Analyst. Mục tiêu cốt lõi của bạn là bảo toàn tài sản, giảm biến động và đảm bảo tăng trưởng ổn định, đáng tin cậy. Bạn ưu tiên sự an toàn, tính bền vững và kiểm soát rủi ro, đồng thời đánh giá kỹ khả năng thua lỗ, suy thoái kinh tế và biến động thị trường. Khi xem xét quyết định/kế hoạch của trader, hãy soi kỹ các thành phần rủi ro cao: chỉ ra nơi quyết định có thể khiến danh mục chịu rủi ro quá mức và nơi phương án thận trọng hơn có thể bảo vệ lợi ích dài hạn. Đây là quyết định của trader:

    {trader_decision}

    Nhiệm vụ của bạn là phản biện chủ động các luận điểm của Risky Analyst và Neutral Analyst, làm rõ nơi họ có thể bỏ sót đe dọa tiềm ẩn hoặc chưa đặt trọng tâm vào tính bền vững. Hãy phản hồi trực tiếp từng điểm của họ, dựa trên các nguồn dữ liệu sau để xây dựng lập luận thuyết phục cho một phương án điều chỉnh theo hướng rủi ro thấp hơn:

    Báo cáo nghiên cứu thị trường: {market_research_report}
    Báo cáo tâm lý mạng xã hội: {sentiment_report}
    Báo cáo thời sự thế giới gần đây: {news_report}
    Báo cáo cơ bản doanh nghiệp: {fundamentals_report}
    Lịch sử hội thoại hiện tại: {history}
    Phản hồi gần nhất của Risky Analyst: {current_risky_response}
    Phản hồi gần nhất của Neutral Analyst: {current_neutral_response}
    Nếu các góc nhìn còn lại chưa có phản hồi, không bịa nội dung; chỉ trình bày lập luận của bạn.

    Hãy chất vấn sự lạc quan quá mức của họ và nhấn mạnh các mặt trái họ có thể bỏ qua. Xử lý từng phản biện để chứng minh vì sao lập trường thận trọng là con đường an toàn nhất cho tài sản của danh mục. Tập trung vào tranh luận và phê bình lập luận để làm nổi bật sức mạnh của chiến lược rủi ro thấp so với các cách tiếp cận khác. Trình bày theo phong cách hội thoại tự nhiên, không cần định dạng đặc biệt."""

        response = llm.invoke(prompt)

        argument = f"Safe Analyst: {response.content}"

        new_risk_debate_state = {
            "history": history + "\n" + argument,
            "risky_history": risk_debate_state.get("risky_history", ""),
            "safe_history": safe_history + "\n" + argument,
            "neutral_history": risk_debate_state.get("neutral_history", ""),
            "alphagpt_history": risk_debate_state.get("alphagpt_history", ""),
            "latest_speaker": "Safe",
            "current_risky_response": risk_debate_state.get(
                "current_risky_response", ""
            ),
            "current_safe_response": argument,
            "current_neutral_response": risk_debate_state.get(
                "current_neutral_response", ""
            ),
            "current_alphagpt_response": risk_debate_state.get("current_alphagpt_response", ""),
            "count": risk_debate_state["count"] + 1,
        }

        return {"risk_debate_state": new_risk_debate_state}

    return safe_node

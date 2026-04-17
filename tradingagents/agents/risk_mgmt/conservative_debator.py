"""
tradingagents/agents/risk_mgmt/conservative_debator.py

FIX: Thêm _s() sanitizer để tránh lỗi 400 JSON malformed khi gửi lên OpenAI API.
"""
import json
from langchain_core.messages import AIMessage

from tradingagents.agents.utils.text_sanitize import sanitize_for_prompt


def create_safe_debator(llm):
    def safe_node(state) -> dict:
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
            alphagpt_context = f"\nTín hiệu định lượng AlphaGPT (tham khảo): {sanitize_for_prompt(current_alphagpt_response)}"

        prompt = (
            "Bạn là Safe/Conservative Risk Analyst. Mục tiêu cốt lõi của bạn là bảo toàn tài sản, "
            "giảm biến động và đảm bảo tăng trưởng ổn định, đáng tin cậy. Bạn ưu tiên sự an toàn, "
            "tính bền vững và kiểm soát rủi ro, đồng thời đánh giá kỹ khả năng thua lỗ, suy thoái "
            "kinh tế và biến động thị trường. Khi xem xét quyết định/kế hoạch của trader, hãy soi kỹ "
            "các thành phần rủi ro cao: chỉ ra nơi quyết định có thể khiến danh mục chịu rủi ro quá "
            "mức và nơi phương án thận trọng hơn có thể bảo vệ lợi ích dài hạn. Đây là quyết định:\n\n"
            f"    {sanitize_for_prompt(trader_decision)}\n"
            f"    {sanitize_for_prompt(alphagpt_context)}\n\n"
            "Nhiệm vụ của bạn là phản biện chủ động các luận điểm của Risky Analyst và Neutral Analyst, "
            "làm rõ nơi họ có thể bỏ sót đe dọa tiềm ẩn hoặc chưa đặt trọng tâm vào tính bền vững. "
            "Nếu tín hiệu AlphaGPT cho thấy hướng short, đây là bằng chứng định lượng hỗ trợ quan "
            "điểm thận trọng của bạn. Hãy dùng dữ liệu sau:\n\n"
            f"Báo cáo nghiên cứu thị trường: {sanitize_for_prompt(market_research_report)}\n"
            f"Báo cáo tâm lý mạng xã hội: {sanitize_for_prompt(sentiment_report)}\n"
            f"Báo cáo thời sự thế giới gần đây: {sanitize_for_prompt(news_report)}\n"
            f"Báo cáo cơ bản doanh nghiệp: {sanitize_for_prompt(fundamentals_report)}\n"
            f"Lịch sử hội thoại hiện tại: {sanitize_for_prompt(history)}\n"
            f"Phản hồi gần nhất của Risky Analyst: {sanitize_for_prompt(current_risky_response)}\n"
            f"Phản hồi gần nhất của Neutral Analyst: {sanitize_for_prompt(current_neutral_response)}\n"
            "Nếu các góc nhìn còn lại chưa có phản hồi, không bịa nội dung; chỉ trình bày lập luận "
            "của bạn.\n\n"
            "Hãy chất vấn sự lạc quan quá mức của họ và nhấn mạnh các mặt trái họ có thể bỏ qua. "
            "Xử lý từng phản biện để chứng minh vì sao lập trường thận trọng là con đường an toàn "
            "nhất. Trình bày theo phong cách hội thoại tự nhiên, không cần định dạng đặc biệt."
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
"""
tradingagents/agents/risk_mgmt/aggresive_debator.py

FIX: Thêm _s() sanitizer để tránh lỗi 400 JSON malformed khi gửi lên OpenAI API.
     Xảy ra khi report chứa null bytes hoặc control characters từ dữ liệu crawl.
"""
import json


def _s(value) -> str:
    """Sanitize bất kỳ value nào thành string an toàn cho LLM prompt."""
    if value is None:
        return ""
    if isinstance(value, str):
        return "".join(
            ch for ch in value
            if ord(ch) != 0 and (ord(ch) >= 32 or ch in ("\n", "\t", "\r"))
        ).strip()
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        return str(value)


def create_risky_debator(llm):
    def risky_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")
        risky_history = risk_debate_state.get("risky_history", "")

        current_safe_response    = risk_debate_state.get("current_safe_response", "")
        current_neutral_response = risk_debate_state.get("current_neutral_response", "")
        current_alphagpt_response = risk_debate_state.get("current_alphagpt_response", "")

        market_research_report = state["market_report"]
        sentiment_report       = state["sentiment_report"]
        news_report            = state["news_report"]
        fundamentals_report    = state["fundamentals_report"]
        trader_decision        = state["trader_investment_plan"]

        alphagpt_context = ""
        if current_alphagpt_response:
            alphagpt_context = f"\nTín hiệu định lượng AlphaGPT (tham khảo): {_s(current_alphagpt_response)}"

        prompt = (
            "Bạn là Risky Risk Analyst. Vai trò của bạn là chủ động ủng hộ các cơ hội lợi nhuận cao "
            "đi kèm rủi ro cao, nhấn mạnh chiến lược táo bạo và lợi thế cạnh tranh. Khi đánh giá "
            "quyết định/kế hoạch của trader, hãy tập trung vào upside, tiềm năng tăng trưởng và lợi "
            "ích đột phá, kể cả khi mức rủi ro cao hơn bình thường. Dùng dữ liệu thị trường và phân "
            "tích tâm lý để củng cố lập luận, đồng thời phản biện các góc nhìn đối lập. Cụ thể, hãy "
            "phản hồi trực tiếp từng điểm do phe Conservative và Neutral nêu ra, dùng phản biện dựa "
            "trên dữ liệu và lập luận thuyết phục. Chỉ ra nơi sự thận trọng của họ có thể bỏ lỡ cơ "
            "hội quan trọng hoặc nơi giả định của họ quá bảo thủ. Đây là quyết định của trader:\n\n"
            f"    {_s(trader_decision)}\n"
            f"    {_s(alphagpt_context)}\n\n"
            "Nhiệm vụ của bạn là xây dựng lập luận thuyết phục ủng hộ quyết định của trader bằng cách "
            "chất vấn và phê bình quan điểm Conservative và Neutral, để chứng minh vì sao góc nhìn lợi "
            "nhuận cao là phương án tốt hơn trong bối cảnh này. Nếu tín hiệu AlphaGPT có hướng long, "
            "hãy dùng làm bằng chứng định lượng bổ sung. Hãy tích hợp insight từ các nguồn sau:\n\n"
            f"Báo cáo nghiên cứu thị trường: {_s(market_research_report)}\n"
            f"Báo cáo tâm lý mạng xã hội: {_s(sentiment_report)}\n"
            f"Báo cáo thời sự thế giới gần đây: {_s(news_report)}\n"
            f"Báo cáo cơ bản doanh nghiệp: {_s(fundamentals_report)}\n"
            f"Lịch sử hội thoại hiện tại: {_s(history)}\n"
            f"Luận điểm gần nhất của Conservative Analyst: {_s(current_safe_response)}\n"
            f"Luận điểm gần nhất của Neutral Analyst: {_s(current_neutral_response)}\n"
            "Nếu các góc nhìn còn lại chưa có phản hồi, không bịa nội dung; chỉ trình bày lập luận "
            "của bạn.\n\n"
            "Hãy tranh luận chủ động: xử lý trực tiếp các lo ngại cụ thể, chỉ ra điểm yếu trong "
            "logic của họ, và khẳng định lợi ích của việc chấp nhận rủi ro để vượt chuẩn thị trường. "
            "Tập trung vào phản biện và thuyết phục, không chỉ liệt kê dữ liệu. Trình bày theo phong "
            "cách hội thoại tự nhiên, không cần định dạng đặc biệt."
        )

        response = llm.invoke(prompt)
        argument = f"Risky Analyst: {response.content}"

        new_risk_debate_state = {
            "history":                  history + "\n" + argument,
            "risky_history":            risky_history + "\n" + argument,
            "safe_history":             risk_debate_state.get("safe_history", ""),
            "neutral_history":          risk_debate_state.get("neutral_history", ""),
            "alphagpt_history":         risk_debate_state.get("alphagpt_history", ""),
            "latest_speaker":           "Risky",
            "current_risky_response":   argument,
            "current_safe_response":    risk_debate_state.get("current_safe_response", ""),
            "current_neutral_response": risk_debate_state.get("current_neutral_response", ""),
            "current_alphagpt_response": current_alphagpt_response,
            "count":                    risk_debate_state["count"] + 1,
        }
        return {"risk_debate_state": new_risk_debate_state}

    return risky_node
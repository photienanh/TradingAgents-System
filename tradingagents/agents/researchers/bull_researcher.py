"""
tradingagents/agents/researchers/bull_researcher.py
"""

from tradingagents.agents.utils.text_sanitize import sanitize_for_prompt


def create_bull_researcher(llm):
    def bull_node(state) -> dict:
        horizon                 = state.get("trading_horizon", "short")
        investment_debate_state = state["investment_debate_state"]
        history                 = investment_debate_state.get("history", "")
        bull_history            = investment_debate_state.get("bull_history", "")
        current_response        = investment_debate_state.get("current_response", "")

        market_research_report  = state["market_report"]
        sentiment_report        = state["sentiment_report"]
        news_report             = state["news_report"]
        fundamentals_report     = state["fundamentals_report"]


        if horizon == "short":
            horizon_instruction = (
                "Đây là tranh luận về cơ hội NGẮN HẠN (2-5 ngày). "
                "Tập trung vào các bằng chứng cho thấy giá có khả năng tăng trong khung thời gian này hoặc bất kỳ tín hiệu nào từ dữ liệu mà bạn cho là có giá trị. "
                "Không cần lập luận về giá trị dài hạn nếu không liên quan đến diễn biến ngắn hạn."
            )
            quant_report = state.get("quant_report", "")
            quant_section = (
                f"Dữ liệu định lượng từ Alpha:\n"
                f"{sanitize_for_prompt(quant_report)}\n\n"
            ) if quant_report else ""
        else:
            horizon_instruction = (
                "Đây là tranh luận về cơ hội ĐẦU TƯ DÀI HẠN. "
                "Tập trung vào các bằng chứng cho thấy doanh nghiệp có nền tảng tốt, "
                "tiềm năng tăng trưởng thực sự, hoặc định giá đang ở mức hấp dẫn. "
                "Dùng bất kỳ góc độ phân tích nào từ dữ liệu mà bạn thấy thuyết phục nhất."
            )
            quant_section = ""

        prompt = f"""Bạn là Bull Analyst trong cuộc tranh luận đầu tư. Nhiệm vụ của bạn là xây dựng lập luận BUY (mua) thuyết phục nhất có thể, dựa trên bằng chứng từ dữ liệu được cung cấp.

## Dữ liệu đã có
Báo cáo thị trường: {sanitize_for_prompt(market_research_report)}
Tâm lý mạng xã hội: {sanitize_for_prompt(sentiment_report)}
Tin tức: {sanitize_for_prompt(news_report)}
Tài chính doanh nghiệp: {sanitize_for_prompt(fundamentals_report)}
{quant_section}
## Lịch sử tranh luận
{sanitize_for_prompt(history)}

## Luận điểm Bear gần nhất (cần phản biện)
{sanitize_for_prompt(current_response)}
---

**Yêu cầu**: {horizon_instruction}
Đọc toàn bộ dữ liệu, xác định những bằng chứng nào ủng hộ mạnh nhất cho luận điểm BUY và trình bày lập luận của bạn. Phản biện trực tiếp từng điểm của Bear - chỉ ra giả định nào đang được diễn giải quá bi quan hoặc thiếu cơ sở. Thừa nhận điểm yếu nếu có - lập luận trung thực có sức thuyết phục hơn lập luận phủ nhận tất cả rủi ro.

Viết theo phong cách hội thoại tranh luận tự nhiên."""

        response = llm.invoke(prompt)
        argument = f"Bull Analyst: {response.content}"

        new_investment_debate_state = {
            "history":          history + "\n" + argument,
            "bull_history":     bull_history + "\n" + argument,
            "bear_history":     investment_debate_state.get("bear_history", ""),
            "current_response": argument,
            "count":            investment_debate_state["count"] + 1,
        }
        return {"investment_debate_state": new_investment_debate_state}

    return bull_node
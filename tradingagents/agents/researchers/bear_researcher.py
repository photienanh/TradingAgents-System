"""
tradingagents/agents/researchers/bear_researcher.py
"""

from tradingagents.agents.utils.text_sanitize import sanitize_for_prompt


def create_bear_researcher(llm):
    def bear_node(state) -> dict:
        horizon                 = state.get("trading_horizon", "short")
        investment_debate_state = state["investment_debate_state"]
        history                 = investment_debate_state.get("history", "")
        bear_history            = investment_debate_state.get("bear_history", "")
        current_response        = investment_debate_state.get("current_response", "")

        market_research_report  = state["market_report"]
        sentiment_report        = state["sentiment_report"]
        news_report             = state["news_report"]
        fundamentals_report     = state["fundamentals_report"]


        if horizon == "short":
            horizon_instruction = (
                "Đây là tranh luận về cơ hội NGẮN HẠN (2-5 ngày). "
                "Tập trung vào các bằng chứng cho thấy giá có khả năng giảm hoặc không tăng trong khung thời gian này, ví dụ như: "
                "áp lực bán, tín hiệu yếu, tin xấu sắp tác động, hoặc bất kỳ dữ liệu nào bạn cho là đáng lo ngại. "
                "Không cần lập luận về rủi ro dài hạn nếu không liên quan đến diễn biến ngắn hạn."
            )
            quant_report = state.get("quant_report", "")
            quant_section = (
                f"Dữ liệu định lượng từ Alpha:\n"
                f"{sanitize_for_prompt(quant_report)}\n\n"
            ) if quant_report else ""
        else:
            horizon_instruction = (
                "Đây là tranh luận về cơ hội ĐẦU TƯ DÀI HẠN. "
                "Tập trung vào các bằng chứng như: doanh nghiệp có rủi ro cấu trúc, "
                "định giá không hợp lý, tăng trưởng thiếu bền vững, hoặc bất kỳ góc độ nào từ dữ liệu "
                "mà bạn thấy là luận điểm AVOID thuyết phục nhất."
            )
            quant_section = ""

        prompt = f"""Bạn là Bear Analyst trong cuộc tranh luận đầu tư. Nhiệm vụ của bạn là xây dựng lập luận SHORT/AVOID (bán hoặc không mua) thuyết phục nhất có thể, dựa trên bằng chứng từ dữ liệu được cung cấp.

## Dữ liệu đã có
Báo cáo thị trường: {sanitize_for_prompt(market_research_report)}
Tâm lý mạng xã hội: {sanitize_for_prompt(sentiment_report)}
Tin tức: {sanitize_for_prompt(news_report)}
Tài chính doanh nghiệp: {sanitize_for_prompt(fundamentals_report)}
{quant_section}
## Lịch sử tranh luận
{sanitize_for_prompt(history)}

## Luận điểm Bull gần nhất (cần phản biện)
{sanitize_for_prompt(current_response)}

---

**Yêu cầu**: {horizon_instruction}

Đọc toàn bộ dữ liệu, xác định những bằng chứng nào ủng hộ mạnh nhất cho luận điểm AVOID/SHORT, và trình bày lập luận của bạn. Phản biện trực tiếp các điểm Bull đã nêu - chỉ ra luận điểm nào đang được diễn giải quá lạc quan hoặc thiếu cơ sở. Thừa nhận điểm mạnh của Bull khi có - lập luận trung thực có sức thuyết phục hơn lập luận phủ nhận tất cả.

Viết theo phong cách hội thoại tranh luận tự nhiên."""

        response = llm.invoke(prompt)
        argument = f"Bear Analyst: {response.content}"

        new_investment_debate_state = {
            "history":          history + "\n" + argument,
            "bear_history":     bear_history + "\n" + argument,
            "bull_history":     investment_debate_state.get("bull_history", ""),
            "current_response": argument,
            "count":            investment_debate_state["count"] + 1,
        }
        return {"investment_debate_state": new_investment_debate_state}

    return bear_node
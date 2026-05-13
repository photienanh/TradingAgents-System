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
                "Tập trung rủi ro NGẮN HẠN: tín hiệu kỹ thuật yếu, volume giảm, "
                "tin xấu sắp tác động, hoặc resistance mạnh phía trên. "
                "Downside trong 2-5 ngày phải cụ thể và có bằng chứng."
            )
            quant_report = state.get("quant_report", "")
            quant_section = (
                f"Dữ liệu định lượng từ Alpha:\n"
                f"{sanitize_for_prompt(quant_report)}\n\n"
            ) if quant_report else ""
        else:
            horizon_instruction = (
                "Tập trung rủi ro DÀI HẠN: định giá quá cao, tăng trưởng chậm lại, "
                "rủi ro ngành, nợ xấu, hoặc lợi thế cạnh tranh đang suy yếu. "
                "Biến động ngắn hạn không phải luận điểm chính."
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
Xây dựng lập luận SHORT/AVOID mạnh nhất có thể. Tập trung vào:
- Rủi ro cụ thể và có bằng chứng (không phải lo ngại chung chung)
- Điểm yếu trong lập luận của Bull: giả định nào là sai, số liệu nào đang bị diễn giải lạc quan quá mức
- Tín hiệu tiêu cực từ dữ liệu tài chính, xu hướng thị trường, hoặc tin bất lợi gần đây
- Chi phí cơ hội: tại sao capital nên được deploy ở nơi khác

Phản biện trực tiếp từng điểm của Bull. Thừa nhận điểm mạnh của họ nếu có — lập luận trung thực có sức thuyết phục hơn.

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
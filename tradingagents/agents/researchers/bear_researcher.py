from langchain_core.messages import AIMessage
import time
import json

from tradingagents.agents.utils.text_sanitize import sanitize_for_prompt


def create_bear_researcher(llm, memory):
    def bear_node(state) -> dict:
        investment_debate_state = state["investment_debate_state"]
        history = investment_debate_state.get("history", "")
        bear_history = investment_debate_state.get("bear_history", "")

        current_response = investment_debate_state.get("current_response", "")
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        for i, rec in enumerate(past_memories, 1):
            past_memory_str += rec["recommendation"] + "\n\n"

        prompt = f"""Bạn là Bear Analyst trong cuộc tranh luận đầu tư. Nhiệm vụ của bạn là xây dựng lập luận SHORT/AVOID (bán hoặc không mua) thuyết phục nhất có thể, dựa trên bằng chứng từ dữ liệu được cung cấp.

## Dữ liệu có sẵn
Báo cáo thị trường: {sanitize_for_prompt(market_research_report)}
Tâm lý mạng xã hội: {sanitize_for_prompt(sentiment_report)}
Tin tức: {sanitize_for_prompt(news_report)}
Tài chính doanh nghiệp: {sanitize_for_prompt(fundamentals_report)}

## Lịch sử tranh luận
{sanitize_for_prompt(history)}

## Luận điểm Bull gần nhất (cần phản biện)
{sanitize_for_prompt(current_response)}

## Bài học quá khứ
{sanitize_for_prompt(past_memory_str)}

---

**Yêu cầu**: Xây dựng lập luận SHORT/AVOID mạnh nhất có thể. Tập trung vào:
- Rủi ro cụ thể và có bằng chứng (không phải lo ngại chung chung)
- Điểm yếu trong lập luận của Bull: giả định nào là sai, số liệu nào đang bị diễn giải lạc quan quá mức
- Tín hiệu tiêu cực: Dùng bằng chứng từ dữ liệu tài chính, xu hướng thị trường, hoặc tin bất lợi gần đây để củng cố lập luận.
- Chi phí cơ hội: tại sao capital nên được deploy ở nơi khác

Phản biện trực tiếp từng điểm của Bull. Thừa nhận điểm mạnh của họ nếu có — lập luận trung thực có sức thuyết phục hơn.

Viết theo phong cách hội thoại tranh luận tự nhiên."""

        response = llm.invoke(prompt)

        argument = f"Bear Analyst: {response.content}"

        new_investment_debate_state = {
            "history": history + "\n" + argument,
            "bear_history": bear_history + "\n" + argument,
            "bull_history": investment_debate_state.get("bull_history", ""),
            "current_response": argument,
            "count": investment_debate_state["count"] + 1,
        }

        return {"investment_debate_state": new_investment_debate_state}

    return bear_node
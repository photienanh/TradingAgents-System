"""
tradingagents/agents/researchers/bull_researcher.py
"""

from tradingagents.agents.utils.text_sanitize import sanitize_for_prompt


def create_bull_researcher(llm, memory):
    def bull_node(state) -> dict:
        investment_debate_state = state["investment_debate_state"]
        history          = investment_debate_state.get("history", "")
        bull_history     = investment_debate_state.get("bull_history", "")
        current_response = investment_debate_state.get("current_response", "")

        market_research_report = state["market_report"]
        sentiment_report       = state["sentiment_report"]
        news_report            = state["news_report"]
        fundamentals_report    = state["fundamentals_report"]
        quant_report = state.get("quant_report", "Không có dữ liệu quant.")

        curr_situation = (
            f"{market_research_report}\n\n{sentiment_report}\n\n"
            f"{news_report}\n\n{fundamentals_report}"
        )
        past_memories = memory.get_memories(curr_situation, n_matches=2)
        past_memory_str = "".join(r["recommendation"] + "\n\n" for r in past_memories)

        prompt = f"""Bạn là Bull Analyst trong cuộc tranh luận đầu tư. Nhiệm vụ của bạn là xây dựng lập luận BUY (mua) thuyết phục nhất có thể, dựa trên bằng chứng từ dữ liệu được cung cấp.

## Dữ liệu định lượng từ AlphaGPT
{sanitize_for_prompt(quant_report)}

IC_OOS và Sharpe_OOS là các chỉ số định lượng. Nếu IC_OOS đủ cao (trên 0.1 là rất tốt), đây là bằng chứng mạnh mẽ. Nếu chỉ số không đủ tốt, tập trung vào phân tích định tính.

## Dữ liệu định tính
Báo cáo thị trường: {sanitize_for_prompt(market_research_report)}
Tâm lý mạng xã hội: {sanitize_for_prompt(sentiment_report)}
Tin tức: {sanitize_for_prompt(news_report)}
Tài chính doanh nghiệp: {sanitize_for_prompt(fundamentals_report)}

## Lịch sử tranh luận
{sanitize_for_prompt(history)}

## Luận điểm Bear gần nhất (cần phản biện)
{sanitize_for_prompt(current_response)}

## Bài học quá khứ
{sanitize_for_prompt(past_memory_str)}

---

**Yêu cầu**: Xây dựng lập luận BUY mạnh nhất có thể. Dùng bằng chứng cụ thể từ dữ liệu. Phản biện trực tiếp từng điểm của Bear. Thừa nhận điểm yếu nếu có — lập luận trung thực có sức thuyết phục hơn lập luận không thừa nhận hạn chế.

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
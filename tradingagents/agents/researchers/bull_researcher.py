"""
tradingagents/agents/researchers/bull_researcher.py
Bull-side researcher node.
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

        prompt = f"""Bạn là Bull Analyst, có nhiệm vụ ủng hộ phương án đầu tư vào cổ phiếu này.

## Dữ liệu định lượng từ Quant Analyst (AlphaGPT)
{sanitize_for_prompt(quant_report)}

Dữ liệu trên là kết quả backtest thống kê nghiêm ngặt trên out-of-sample data.
IC_OOS và Sharpe_OOS là các chỉ số định lượng — hãy sử dụng chúng như
bằng chứng cứng khi lập luận.

## Dữ liệu định tính
Báo cáo thị trường: {sanitize_for_prompt(market_research_report)}
Báo cáo tâm lý mạng xã hội: {sanitize_for_prompt(sentiment_report)}
Tin tức thế giới gần đây: {sanitize_for_prompt(news_report)}
Báo cáo cơ bản doanh nghiệp: {sanitize_for_prompt(fundamentals_report)}

## Lịch sử tranh luận
{sanitize_for_prompt(history)}

## Luận điểm Bear gần nhất
{sanitize_for_prompt(current_response)}

## Phản tư từ quá khứ
{sanitize_for_prompt(past_memory_str)}

---

Nhiệm vụ của bạn:
1. Xây dựng lập luận Bull dựa trên CẢ HAI nguồn: quant signal (IC_OOS, Sharpe, 
   hướng alpha) VÀ phân tích định tính (market, fundamentals, news, sentiment).
2. Nếu quant signal ủng hộ LONG, hãy trích dẫn cụ thể alpha nào và con số IC_OOS.
3. Nếu quant signal mâu thuẫn với định tính, hãy giải thích tại sao định tính 
   quan trọng hơn trong trường hợp này (hoặc ngược lại).
4. Phản biện trực tiếp từng điểm của Bear Analyst.

Trình bày theo phong cách hội thoại tự nhiên."""

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
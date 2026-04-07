"""
tradingagents/agents/researchers/bull_researcher.py
Bull-side researcher node.
"""

from langchain_core.messages import AIMessage


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
{quant_report}

Dữ liệu trên là kết quả backtest thống kê nghiêm ngặt trên out-of-sample data.
IC_OOS và Sharpe_OOS là các chỉ số KHÔNG bị overfit — hãy sử dụng chúng như
bằng chứng cứng khi lập luận.

## Dữ liệu định tính
Báo cáo thị trường: {market_research_report}
Báo cáo tâm lý mạng xã hội: {sentiment_report}
Tin tức thế giới gần đây: {news_report}
Báo cáo cơ bản doanh nghiệp: {fundamentals_report}

## Lịch sử tranh luận
{history}

## Luận điểm Bear gần nhất
{current_response}

## Phản tư từ quá khứ
{past_memory_str}

---

Nhiệm vụ của bạn:
1. Xây dựng lập luận Bull dựa trên CẢ HAI nguồn: quant signal (IC_OOS, Sharpe, 
   hướng alpha) VÀ phân tích định tính (fundamentals, news, sentiment).
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


# ─────────────────────────────────────────────────────────────────────


def create_bear_researcher(llm, memory):
    def bear_node(state) -> dict:
        investment_debate_state = state["investment_debate_state"]
        history          = investment_debate_state.get("history", "")
        bear_history     = investment_debate_state.get("bear_history", "")
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

        prompt = f"""Bạn là Bear Analyst, có nhiệm vụ phân tích rủi ro và lập luận thận trọng.

## Dữ liệu định lượng từ Quant Analyst (AlphaGPT)
{quant_report}

Dữ liệu trên là kết quả backtest thống kê trên out-of-sample data.
Hãy chú ý đặc biệt đến phần CẢNH BÁO trong report — overfit flag, flip flag,
turnover cao, IC_OOS thấp — đây là những điểm yếu của quant model.

## Dữ liệu định tính
Báo cáo thị trường: {market_research_report}
Báo cáo tâm lý mạng xã hội: {sentiment_report}
Tin tức thế giới gần đây: {news_report}
Báo cáo cơ bản doanh nghiệp: {fundamentals_report}

## Lịch sử tranh luận
{history}

## Luận điểm Bull gần nhất
{current_response}

## Phản tư từ quá khứ
{past_memory_str}

---

Nhiệm vụ của bạn:
1. Xây dựng lập luận Bear dựa trên CẢ HAI nguồn.
2. Nếu quant signal có cảnh báo (flip, overfit, IC thấp), hãy khai thác 
   những điểm yếu đó để làm suy yếu luận điểm Bull.
3. Nếu quant signal mạnh nhưng định tính yếu, hãy giải thích tại sao 
   context thị trường hiện tại làm mô hình lịch sử kém đáng tin.
4. Phản biện trực tiếp từng điểm của Bull Analyst.

Trình bày theo phong cách hội thoại tự nhiên."""

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
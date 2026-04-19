import time
import json

from tradingagents.agents.utils.text_sanitize import sanitize_for_prompt


def create_research_manager(llm, memory):
    def research_manager_node(state) -> dict:
        history = state["investment_debate_state"].get("history", "")
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        investment_debate_state = state["investment_debate_state"]

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        for i, rec in enumerate(past_memories, 1):
            past_memory_str += rec["recommendation"] + "\n\n"

        prompt = f"""Bạn là Research Manager — người tổng hợp kết quả tranh luận và đưa ra quyết định đầu tư.

## NGUYÊN TẮC TRUNG LẬP (BẮT BUỘC)
Quyết định BUY, SELL và HOLD đều có giá trị ngang nhau. Không có hướng nào là "mặc định" hay "an toàn hơn". Chỉ dữ liệu và lập luận quyết định kết quả.

## TIÊU CHUẨN ĐƯA RA QUYẾT ĐỊNH
- **BUY**: Bull có bằng chứng cụ thể, thuyết phục hơn Bear về tiềm năng tăng giá
- **SELL**: Bear có bằng chứng cụ thể, thuyết phục hơn Bull về rủi ro giảm giá hoặc cơ hội cơ hội chi phí
- **HOLD**: Bằng chứng hai bên cân bằng nhau, chưa có catalyst rõ ràng để hành động

## QUY TRÌNH ĐÁNH GIÁ
1. Xác định luận điểm mạnh nhất của Bull và luận điểm mạnh nhất của Bear
2. Đánh giá chất lượng bằng chứng: dữ liệu cụ thể > ý kiến chủ quan
3. Xét bài học từ các tình huống tương tự trong quá khứ
4. Đưa ra quyết định dựa trên bên có bằng chứng thuyết phục hơn

Bài học từ quá khứ:
"{sanitize_for_prompt(past_memory_str)}"

Lịch sử tranh luận:
{sanitize_for_prompt(history)}

## YÊU CẦU OUTPUT (BẮT BUỘC tuân theo format)

### Quyết Định Nhóm Nghiên Cứu — {{ticker}} — {{date}}

#### Tóm Tắt Tranh Luận
**Bull:** [Luận điểm mạnh nhất, bằng chứng cụ thể]
**Bear:** [Luận điểm mạnh nhất, bằng chứng cụ thể]

#### Đánh Giá
[2-3 câu phân tích tại sao một bên thuyết phục hơn, hoặc tại sao bằng chứng cân bằng]

#### Quyết Định: **[BUY / SELL / HOLD]**

#### Kế Hoạch Hành Động
- Lý do chính: [1-2 lý do quyết định]
- Điều kiện để xem xét lại: [Điều gì sẽ thay đổi quyết định này]
- Rủi ro chính cần theo dõi: [1-2 rủi ro quan trọng nhất]"""

        # Format ticker and date into the prompt
        ticker = state.get("company_of_interest", "N/A")
        trade_date = state.get("trade_date", "N/A")
        prompt = prompt.replace("{ticker}", ticker).replace("{date}", trade_date)

        response = llm.invoke(prompt)

        new_investment_debate_state = {
            "judge_decision": response.content,
            "history": investment_debate_state.get("history", ""),
            "bear_history": investment_debate_state.get("bear_history", ""),
            "bull_history": investment_debate_state.get("bull_history", ""),
            "current_response": response.content,
            "count": investment_debate_state["count"],
        }

        return {
            "investment_debate_state": new_investment_debate_state,
            "investment_plan": response.content,
        }

    return research_manager_node
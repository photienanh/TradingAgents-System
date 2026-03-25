import time
import json


def create_risk_manager(llm, memory):
    def risk_manager_node(state) -> dict:

        company_name = state["company_of_interest"]

        history = state["risk_debate_state"]["history"]
        risk_debate_state = state["risk_debate_state"]
        market_research_report = state["market_report"]
        news_report = state["news_report"]
        fundamentals_report = state["news_report"]
        sentiment_report = state["sentiment_report"]
        trader_plan = state["investment_plan"]

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        for i, rec in enumerate(past_memories, 1):
            past_memory_str += rec["recommendation"] + "\n\n"

        prompt = f"""Với vai trò Trọng tài Quản trị Rủi ro và Điều phối tranh luận, nhiệm vụ của bạn là đánh giá cuộc tranh luận giữa ba nhà phân tích rủi ro (Risky, Neutral, Safe/Conservative) và đưa ra phương án tốt nhất cho trader. Quyết định cuối cùng phải rõ ràng: Buy, Sell hoặc Hold. Chỉ chọn Hold khi có lập luận thật sự thuyết phục, không dùng Hold như phương án an toàn mặc định. Hãy ưu tiên tính rõ ràng và dứt khoát.

    Nguyên tắc ra quyết định:
    1. **Tóm tắt luận điểm chính**: Trích xuất các điểm mạnh nhất từ từng nhà phân tích, bám sát bối cảnh hiện tại.
    2. **Nêu cơ sở lập luận**: Bảo vệ khuyến nghị bằng các luận điểm và phản biện cụ thể từ cuộc tranh luận.
    3. **Tinh chỉnh kế hoạch giao dịch**: Bắt đầu từ kế hoạch ban đầu của trader, **{trader_plan}**, rồi điều chỉnh theo các insight từ nhóm phân tích.
    4. **Học từ sai lầm trước đó**: Dùng các bài học trong **{past_memory_str}** để tránh lặp lại lỗi cũ và cải thiện chất lượng quyết định BUY/SELL/HOLD, giảm rủi ro thua lỗ.

    Đầu ra cần có:
    - Một khuyến nghị rõ ràng, có thể hành động ngay: Buy, Sell hoặc Hold.
    - Phần lập luận chi tiết, bám sát nội dung tranh luận và các phản tư trong quá khứ.

    ---

    **Lịch sử tranh luận của các nhà phân tích:**  
    {history}

    ---

    Hãy tập trung vào insight có thể triển khai, cải tiến liên tục, và đánh giá nghiêm túc mọi góc nhìn để đảm bảo quyết định cuối cùng mang lại kết quả tốt hơn."""

        response = llm.invoke(prompt)

        new_risk_debate_state = {
            "judge_decision": response.content,
            "history": risk_debate_state["history"],
            "risky_history": risk_debate_state["risky_history"],
            "safe_history": risk_debate_state["safe_history"],
            "neutral_history": risk_debate_state["neutral_history"],
            "alphagpt_history": risk_debate_state.get("alphagpt_history", ""),
            "latest_speaker": "Judge",
            "current_risky_response": risk_debate_state["current_risky_response"],
            "current_safe_response": risk_debate_state["current_safe_response"],
            "current_neutral_response": risk_debate_state["current_neutral_response"],
            "current_alphagpt_response": risk_debate_state.get("current_alphagpt_response", ""),
            "count": risk_debate_state["count"],
        }

        return {
            "risk_debate_state": new_risk_debate_state,
            "final_trade_decision": response.content,
        }

    return risk_manager_node

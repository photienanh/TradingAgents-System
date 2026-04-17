"""
tradingagents/agents/managers/risk_manager.py
"""
import time
from tradingagents.agents.utils.text_sanitize import sanitize_for_prompt

def create_risk_manager(llm, memory):
    def risk_manager_node(state) -> dict:

        company_name = state["company_of_interest"]
        risk_debate_state = state["risk_debate_state"]
        history = state["risk_debate_state"]["history"]
        market_research_report = state["market_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]
        sentiment_report = state["sentiment_report"]
        trader_plan = state["investment_plan"]

        # Lấy AlphaGPT signal context
        alphagpt_response = risk_debate_state.get("current_alphagpt_response", "")
        alphagpt_context = ""
        if alphagpt_response:
            alphagpt_context = f"\n\n**Tín hiệu định lượng từ AlphaGPT:**\n{alphagpt_response}"

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        for i, rec in enumerate(past_memories, 1):
            past_memory_str += rec["recommendation"] + "\n\n"

        prompt = prompt = (
            f"Bạn là Risk Manager — trọng tài cuối cùng đưa ra quyết định giao dịch cho {company_name}. "
            "Nhiệm vụ của bạn là đánh giá tranh luận từ ba analyst (Risky, Neutral, Safe) và đưa ra "
            "khuyến nghị BUY, SELL hoặc HOLD.\n\n"
 
            "**NGUYÊN TẮC QUAN TRỌNG — ĐỌC KỸ TRƯỚC KHI QUYẾT ĐỊNH:**\n"
            "1. Không có bias mặc định về hướng nào. BUY, SELL và HOLD đều hợp lệ như nhau — "
            "quyết định phải dựa trên bằng chứng, không phải thói quen.\n"
            "2. Safe Analyst thường phóng đại rủi ro. Hãy đánh giá xem lo ngại của họ có "
            "THỰC SỰ ảnh hưởng đến giá trong ngắn-trung hạn không, hay chỉ là rủi ro lý thuyết.\n"
            "3. AlphaGPT signal BUY/SELL rõ ràng (z-score > 1.0 hoặc < -1.0) là "
            "bằng chứng định lượng khá đáng tin cậy.\n"
            "4. SELL chỉ hợp lý khi có catalyst tiêu cực rõ ràng (kết quả kinh doanh xấu, "
            "rủi ro hệ thống, tin tức có thể ảnh hưởng trực tiếp đến công ty). Không SELL chỉ vì "
            "thị trường 'không chắc chắn'.\n\n"
 
            f"{alphagpt_context}\n\n"
 
            "**Kế hoạch ban đầu của Trader:**\n"
            f"{sanitize_for_prompt(trader_plan)}\n\n"
 
            "**Bài học từ quyết định trước:**\n"
            f"{sanitize_for_prompt(past_memory_str)}\n\n"
 
            "**Toàn bộ lịch sử tranh luận:**\n"
            f"{sanitize_for_prompt(history)}\n\n"
 
            "---\n"
            "**QUY TRÌNH RA QUYẾT ĐỊNH:**\n"
            "Bước 1: Xác định luận điểm MẠNH NHẤT của Risky và MẠNH NHẤT của Safe. "
            "Bên nào có bằng chứng cụ thể hơn (số liệu, catalyst, xu hướng rõ ràng)?\n"
            "Bước 2: Đối chiếu với tín hiệu AlphaGPT nếu có. "
            "Signal định lượng có nhất quán với phân tích định tính không?\n"
            "Bước 3: Đưa ra quyết định. Nếu một bên thuyết phục hơn → theo bên đó."
            "Nếu bằng chứng BUY và SELL cân nhau → HOLD. \n\n"
 
            "Trả lời theo format:\n"
            "**QUYẾT ĐỊNH: [BUY/SELL/HOLD]**\n"
            "**Lý do:** [2-3 câu tóm tắt căn cứ chính]\n"
            "**Chi tiết:** [phân tích đầy đủ dựa trên tranh luận và tín hiệu]"
        )

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
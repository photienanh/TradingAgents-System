"""
tradingagents/agents/managers/risk_manager.py
"""
import time
from tradingagents.agents.utils.text_sanitize import sanitize_for_prompt

def create_risk_manager(llm, memory):
    def risk_manager_node(state) -> dict:

        company_name = state["company_of_interest"]
        trade_date = state.get("trade_date", "N/A")
        risk_debate_state = state["risk_debate_state"]
        history = state["risk_debate_state"]["history"]
        market_research_report = state["market_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]
        sentiment_report = state["sentiment_report"]
        trader_plan = state["investment_plan"]

        alphagpt_response = risk_debate_state.get("current_alphagpt_response", "")
        alphagpt_context = ""
        if alphagpt_response:
            alphagpt_context = f"\n\n**Tín hiệu định lượng từ AlphaGPT:**\n{sanitize_for_prompt(alphagpt_response)}"

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        for i, rec in enumerate(past_memories, 1):
            past_memory_str += rec["recommendation"] + "\n\n"

        prompt = (
            f"Bạn là Portfolio Manager — người ra quyết định giao dịch cuối cùng cho {company_name}.\n\n"

            "## NGUYÊN TẮC TRUNG LẬP (BẮT BUỘC)\n"
            "BUY, SELL và HOLD đều là các quyết định hợp lệ và có giá trị ngang nhau.\n"
            "Nhiệm vụ của bạn là chọn quyết định có bằng chứng thuyết phục nhất — không phải quyết định 'an toàn nhất'.\n\n"

            "## TIÊU CHUẨN ĐÁNH GIÁ\n"
            "Đánh giá bằng chứng từ ba analyst theo trọng số:\n"
            "- **Risky Analyst**: Phát hiện upside, cơ hội bị bỏ lỡ do quá thận trọng\n"
            "- **Safe Analyst**: Phát hiện downside, rủi ro bị đánh giá thấp\n"
            "- **Neutral Analyst**: Cân bằng hai bên, chỉ ra điểm yếu của cả hai\n\n"
            "**Lưu ý quan trọng**: Mỗi analyst có xu hướng cố hữu — Risky luôn lạc quan, Safe luôn lo ngại. "
            "Hãy đánh giá CHẤT LƯỢNG BẰNG CHỨNG, không phải theo chiều hướng của analyst.\n\n"

            "## QUY TRÌNH RA QUYẾT ĐỊNH\n"
            "Bước 1: Xác định luận điểm có bằng chứng cụ thể nhất từ mỗi bên\n"
            "Bước 2: Đối chiếu với tín hiệu AlphaGPT (nếu có) và kế hoạch của Research team\n"
            "Bước 3: Quyết định:\n"
            "   - Nếu bằng chứng LONG rõ ràng hơn → BUY\n"
            "   - Nếu bằng chứng SHORT rõ ràng hơn → SELL\n"
            "   - Nếu bằng chứng hai bên cân bằng → HOLD\n\n"

            f"{alphagpt_context}\n\n"

            f"**Kế hoạch từ Research team:**\n{sanitize_for_prompt(trader_plan)}\n\n"

            f"**Bài học từ quyết định trước:**\n{sanitize_for_prompt(past_memory_str)}\n\n"

            f"**Toàn bộ lịch sử tranh luận:**\n{sanitize_for_prompt(history)}\n\n"

            "---\n"
            f"## YÊU CẦU OUTPUT (BẮT BUỘC tuân theo format)\n\n"

            f"### Quyết Định Cuối Cùng — {company_name} — {trade_date}\n\n"

            "#### Đánh Giá Tranh Luận\n"
            "**Luận điểm mạnh nhất ủng hộ tăng giá:** [từ Risky hoặc bất kỳ analyst]\n"
            "**Luận điểm mạnh nhất ủng hộ giảm/thận trọng:** [từ Safe hoặc bất kỳ analyst]\n"
            "**Tín hiệu AlphaGPT:** [hướng và mức độ tin cậy]\n\n"

            "#### Lý Do Quyết Định\n"
            "[2-3 câu giải thích tại sao bằng chứng dẫn đến quyết định này]\n\n"

            "#### Quyết Định: **[BUY / SELL / HOLD]**\n\n"

            "#### Điều Kiện Xem Xét Lại\n"
            "- Nên đổi sang BUY nếu: [điều kiện cụ thể]\n"
            "- Nên đổi sang SELL nếu: [điều kiện cụ thể]\n\n"

            "#### Quản Lý Rủi Ro\n"
            "- Rủi ro chính: [1-2 rủi ro quan trọng nhất cần theo dõi]\n"
            "- Khung thời gian: [ngắn/trung/dài hạn]"
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
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

        prompt = f"""Với vai trò Quản lý danh mục kiêm Điều phối tranh luận, nhiệm vụ của bạn là đánh giá phản biện một cách chặt chẽ và đưa ra quyết định dứt khoát: nghiêng về Bear, nghiêng về Bull, hoặc chọn Hold khi có căn cứ rất mạnh từ các lập luận đã nêu.

    Hãy tóm tắt ngắn gọn các luận điểm quan trọng của hai phía, tập trung vào bằng chứng/lập luận thuyết phục nhất. Khuyến nghị cuối cùng (Buy, Sell hoặc Hold) phải rõ ràng và có thể hành động. Hãy chốt phương án dựa trên các luận điểm mạnh nhất trong cuộc tranh luận.

    Ngoài ra, hãy xây dựng một kế hoạch đầu tư chi tiết cho trader, bao gồm:

    Khuyến nghị cuối cùng: Quan điểm dứt khoát dựa trên các luận điểm thuyết phục nhất.
    Cơ sở kết luận: Giải thích vì sao các luận điểm đó dẫn đến quyết định của bạn.
    Hành động chiến lược: Các bước cụ thể để triển khai khuyến nghị.
    Hãy tính đến các sai lầm trước đây trong bối cảnh tương tự. Dùng các bài học này để cải thiện chất lượng ra quyết định và thể hiện bạn đang học hỏi liên tục. Trình bày tự nhiên theo văn phong hội thoại, không cần định dạng cầu kỳ.

    Các phản tư sai lầm trong quá khứ:
    \"{sanitize_for_prompt(past_memory_str)}\"

    Nội dung tranh luận:
    Lịch sử tranh luận:
    {sanitize_for_prompt(history)}"""
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

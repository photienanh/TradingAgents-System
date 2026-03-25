import functools
import time
import json


def create_trader(llm, memory):
    def trader_node(state, name):
        company_name = state["company_of_interest"]
        investment_plan = state["investment_plan"]
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        if past_memories:
            for i, rec in enumerate(past_memories, 1):
                past_memory_str += rec["recommendation"] + "\n\n"
        else:
            past_memory_str = "Không có bài học quá khứ phù hợp."

        context = {
            "role": "user",
            "content": f"Dựa trên phân tích tổng hợp từ nhóm analyst, dưới đây là kế hoạch đầu tư dành cho {company_name}. Kế hoạch này tổng hợp insight từ xu hướng kỹ thuật hiện tại, bối cảnh vĩ mô và tâm lý mạng xã hội. Hãy dùng kế hoạch này làm nền tảng để đánh giá quyết định giao dịch tiếp theo.\n\nKế hoạch đầu tư đề xuất: {investment_plan}\n\nHãy tận dụng các insight này để đưa ra quyết định có cơ sở và mang tính chiến lược.",
        }

        messages = [
            {
                "role": "system",
                "content": f"""Bạn là trading agent phân tích dữ liệu thị trường để đưa ra quyết định đầu tư. Dựa trên phân tích của bạn, hãy đưa ra một khuyến nghị cụ thể: buy, sell hoặc hold. Phần kết luận phải rõ ràng và bắt buộc kết thúc bằng đúng cú pháp 'FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**' để xác nhận khuyến nghị. Không được bỏ qua việc học từ các quyết định trước đây để tránh lặp lại sai lầm. Đây là các phản tư từ tình huống tương tự và bài học rút ra: {past_memory_str}""",
            },
            context,
        ]

        result = llm.invoke(messages)

        return {
            "messages": [result],
            "trader_investment_plan": result.content,
            "sender": name,
        }

    return functools.partial(trader_node, name="Trader")

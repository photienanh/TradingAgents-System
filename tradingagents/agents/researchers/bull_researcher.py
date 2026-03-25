from langchain_core.messages import AIMessage
import time
import json


def create_bull_researcher(llm, memory):
    def bull_node(state) -> dict:
        investment_debate_state = state["investment_debate_state"]
        history = investment_debate_state.get("history", "")
        bull_history = investment_debate_state.get("bull_history", "")

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

        prompt = f"""Bạn là Bull Analyst, có nhiệm vụ ủng hộ phương án đầu tư vào cổ phiếu. Hãy xây dựng một lập luận mạnh, dựa trên bằng chứng, nhấn mạnh tiềm năng tăng trưởng, lợi thế cạnh tranh và các tín hiệu tích cực của thị trường. Tận dụng dữ liệu nghiên cứu được cung cấp để phản biện hiệu quả các luận điểm phe Bear.

    Các trọng tâm cần tập trung:
    - Tiềm năng tăng trưởng: Nêu rõ cơ hội thị trường, triển vọng doanh thu và khả năng mở rộng của doanh nghiệp.
    - Lợi thế cạnh tranh: Nhấn mạnh điểm mạnh như sản phẩm khác biệt, thương hiệu vững, hoặc vị thế thị trường vượt trội.
    - Tín hiệu tích cực: Dùng dữ liệu sức khỏe tài chính, xu hướng ngành và tin tức tích cực gần đây để củng cố luận điểm.
    - Phản biện phe Bear: Phân tích kỹ các luận điểm Bear bằng dữ liệu cụ thể và lập luận chặt chẽ; giải thích vì sao góc nhìn Bull thuyết phục hơn.
    - Tính tương tác: Trình bày theo phong cách tranh luận hội thoại, bám sát các điểm của phe Bear thay vì chỉ liệt kê dữ liệu.

    Nguồn dữ liệu có sẵn:
    Báo cáo thị trường: {market_research_report}
    Báo cáo tâm lý mạng xã hội: {sentiment_report}
    Tin tức thế giới gần đây: {news_report}
    Báo cáo cơ bản doanh nghiệp: {fundamentals_report}
    Lịch sử tranh luận: {history}
    Luận điểm Bear gần nhất: {current_response}
    Phản tư từ các tình huống tương tự và bài học rút ra: {past_memory_str}

    Hãy dùng các thông tin trên để tạo lập luận Bull thuyết phục, phản biện hiệu quả các lo ngại của phe Bear và duy trì một cuộc tranh luận sắc nét thể hiện rõ điểm mạnh của góc nhìn Bull. Đồng thời, bạn phải tận dụng phần phản tư để học từ sai lầm trước đây và cải thiện chất lượng lập luận hiện tại.
    """

        response = llm.invoke(prompt)

        argument = f"Bull Analyst: {response.content}"

        new_investment_debate_state = {
            "history": history + "\n" + argument,
            "bull_history": bull_history + "\n" + argument,
            "bear_history": investment_debate_state.get("bear_history", ""),
            "current_response": argument,
            "count": investment_debate_state["count"] + 1,
        }

        return {"investment_debate_state": new_investment_debate_state}

    return bull_node

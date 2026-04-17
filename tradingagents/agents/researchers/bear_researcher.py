from langchain_core.messages import AIMessage
import time
import json

from tradingagents.agents.utils.text_sanitize import sanitize_for_prompt


def create_bear_researcher(llm, memory):
    def bear_node(state) -> dict:
        investment_debate_state = state["investment_debate_state"]
        history = investment_debate_state.get("history", "")
        bear_history = investment_debate_state.get("bear_history", "")

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

        prompt = f"""Bạn là Bear Analyst, có nhiệm vụ lập luận phản đối việc đầu tư vào cổ phiếu này. Mục tiêu của bạn là đưa ra lập luận chặt chẽ, nhấn mạnh rủi ro, thách thức và các tín hiệu tiêu cực. Hãy tận dụng dữ liệu nghiên cứu được cung cấp để làm rõ các mặt bất lợi và phản biện hiệu quả các luận điểm phe Bull.

    Các trọng tâm cần tập trung:

    - Rủi ro và thách thức: Nêu rõ các yếu tố như bão hòa thị trường, bất ổn tài chính, hoặc rủi ro vĩ mô có thể cản trở hiệu quả cổ phiếu.
    - Điểm yếu cạnh tranh: Nhấn mạnh các lỗ hổng như vị thế thị trường yếu hơn, đổi mới suy giảm, hoặc áp lực từ đối thủ.
    - Tín hiệu tiêu cực: Dùng bằng chứng từ dữ liệu tài chính, xu hướng thị trường, hoặc tin bất lợi gần đây để củng cố lập luận.
    - Phản biện phe Bull: Phân tích kỹ lập luận Bull bằng dữ liệu cụ thể và lập luận logic, chỉ ra điểm yếu hoặc giả định quá lạc quan.
    - Tính tương tác: Trình bày theo phong cách tranh luận hội thoại, bám sát các điểm của phe Bull thay vì chỉ liệt kê thông tin.

    Nguồn dữ liệu có sẵn:

    Báo cáo thị trường: {sanitize_for_prompt(market_research_report)}
    Báo cáo tâm lý mạng xã hội: {sanitize_for_prompt(sentiment_report)}
    Tin tức thế giới gần đây: {sanitize_for_prompt(news_report)}
    Báo cáo cơ bản doanh nghiệp: {sanitize_for_prompt(fundamentals_report)}
    Lịch sử tranh luận: {sanitize_for_prompt(history)}
    Luận điểm Bull gần nhất: {sanitize_for_prompt(current_response)}
    Phản tư từ các tình huống tương tự và bài học rút ra: {sanitize_for_prompt(past_memory_str)}

    Hãy dùng các thông tin trên để xây dựng lập luận Bear thuyết phục, phản bác hiệu quả các tuyên bố của phe Bull và duy trì một cuộc tranh luận sắc nét làm rõ rủi ro cũng như điểm yếu của quyết định đầu tư. Đồng thời, bạn phải tận dụng phần phản tư để học từ sai lầm trong quá khứ và cải thiện chất lượng lập luận hiện tại.
    """

        response = llm.invoke(prompt)

        argument = f"Bear Analyst: {response.content}"

        new_investment_debate_state = {
            "history": history + "\n" + argument,
            "bear_history": bear_history + "\n" + argument,
            "bull_history": investment_debate_state.get("bull_history", ""),
            "current_response": argument,
            "count": investment_debate_state["count"] + 1,
        }

        return {"investment_debate_state": new_investment_debate_state}

    return bear_node

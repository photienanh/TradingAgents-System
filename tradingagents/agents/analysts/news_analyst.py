from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json
from tradingagents.agents.utils.agent_utils import get_news, get_global_news
from tradingagents.dataflows.config import get_config


def create_news_analyst(llm):
    def news_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]

        tools = [
            get_news,
            get_global_news,
        ]

        system_message = (
            """Bạn là News Analyst chuyên nghiệp - nhà nghiên cứu tin tức. Nhiệm vụ DUY NHẤT của bạn là thu thập và phân tích tin tức — cả tin tức doanh nghiệp lẫn tin tức vĩ mô. Bạn KHÔNG đưa ra khuyến nghị giao dịch (BUY/SELL/HOLD).

Hãy phân tích toàn diện:
- Tin tức doanh nghiệp trực tiếp:
- Tin tức ngành và đối thủ cạnh tranh
- Tin tức vĩ mô trong nước:
- Tin tức quốc tế:
- Tác động tiềm năng của từng tin đến doanh nghiệp
- Rủi ro tin tức chưa được phản ánh vào giá

Công cụ sử dụng:
- get_news(query, start_date, end_date): cho tin theo công ty/chủ đề
- get_global_news(curr_date, look_back_days, limit): cho tin vĩ mô tổng quát

## Cấu trúc báo cáo (BẮT BUỘC tuân theo)

### Phân Tích Tin Tức — {ticker} — {current_date}

#### 1. Tin Tức Doanh Nghiệp
[Các tin quan trọng trực tiếp liên quan đến công ty trong tuần qua]

#### 2. Tin Tức Ngành
[Xu hướng ngành, tin tức từ đối thủ, đối tác (nếu có)]

#### 3. Bối Cảnh Vĩ Mô Trong Nước
[Chính sách, quy định, kinh tế vĩ mô Việt Nam ảnh hưởng đến doanh nghiệp]

#### 4. Bối Cảnh Quốc Tế
[Tin tức toàn cầu có liên quan: Fed, tỷ giá, giá nguyên liệu, địa chính trị]

#### 5. Đánh Giá Tác Động
[Tổng hợp các tin tức có tác động tích cực/tiêu cực đến doanh nghiệp, phân loại theo mức độ]

| Tin Tức | Loại | Tác Động Tiềm Năng | Mức Độ |
|---------|------|-------------------|--------|
[Bảng tổng hợp tin tức quan trọng]"""
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Bạn là News Analyst đang thu thập và phân tích tin tức."
                    " Hãy dùng các công cụ để lấy tin tức từ nhiều nguồn."
                    " Nhiệm vụ của bạn là cung cấp phân tích tin tức khách quan, đầy đủ — KHÔNG đưa ra khuyến nghị BUY/SELL/HOLD."
                    " Bạn có quyền truy cập: {tool_names}.\n{system_message}"
                    " Ngày hiện tại: {current_date}. Mã cần phân tích: {ticker}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(ticker=ticker)

        chain = prompt | llm.bind_tools(tools)
        result = chain.invoke(state["messages"])

        report = ""
        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "news_report": report,
        }

    return news_analyst_node
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json
from tradingagents.agents.utils.agent_utils import get_news
from tradingagents.dataflows.config import get_config


def create_social_media_analyst(llm):
    def social_media_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]

        tools = [
            get_news,
        ]

        system_message = (
            """Bạn là Social Media & Sentiment Analyst chuyên nghiệp - nhà phân tích mạng xã hội và tin tức theo từng doanh nghiệp. Nhiệm vụ DUY NHẤT của bạn là thu thập và phân tích tâm lý thị trường, thảo luận mạng xã hội và tin tức doanh nghiệp. Bạn KHÔNG đưa ra khuyến nghị giao dịch (BUY/SELL/HOLD).

Hãy phân tích toàn diện:
- Tâm lý thị trường đối với mã cổ phiếu (tích cực/tiêu cực/trung lập, mức độ)
- Các luồng thảo luận chính trên mạng xã hội và diễn đàn đầu tư
- Tin tức doanh nghiệp gần đây và phản ứng thị trường
- Xu hướng sentiment theo thời gian (có thay đổi không?)
- Các sự kiện/catalyst tiềm năng đang được thảo luận
- Mức độ quan tâm và hoạt động của nhà đầu tư cá nhân

Dùng công cụ get_news(query, start_date, end_date) để tìm tin tức và thảo luận liên quan.

## Cấu trúc báo cáo (BẮT BUỘC tuân theo)

### Phân Tích Tâm Lý & Mạng Xã Hội — {ticker} — {current_date}

#### 1. Tổng Quan Tâm Lý
[Đánh giá tổng thể sentiment: tích cực/tiêu cực/trung lập, độ mạnh]

#### 2. Tin Tức Doanh Nghiệp Gần Đây
[Các tin tức quan trọng, phản ứng thị trường]

#### 3. Thảo Luận Mạng Xã Hội
[Các chủ đề đang được thảo luận, quan điểm phổ biến]

#### 4. Catalyst Tiềm Năng
[Các sự kiện/tin tức có thể ảnh hưởng đến giá trong tương lai gần]

#### 5. Rủi Ro Từ Sentiment
[Các rủi ro từ tâm lý thị trường: tin đồn, overreaction, FOMO/panic]

| Khía Cạnh | Đánh Giá | Mức Độ Ảnh Hưởng |
|-----------|---------|-----------------|
[Bảng tổng hợp các điểm chính]"""
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Bạn là Social Media & Sentiment Analyst đang thu thập dữ liệu tâm lý thị trường."
                    " Hãy dùng các công cụ để tìm tin tức và thảo luận liên quan."
                    " Nhiệm vụ của bạn là cung cấp phân tích sentiment khách quan — KHÔNG đưa ra khuyến nghị BUY/SELL/HOLD."
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
            "sentiment_report": report,
        }

    return social_media_analyst_node
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tradingagents.agents.utils.agent_utils import get_f247_forum_posts, get_ticker_news


def create_social_media_analyst(llm):
    def social_media_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]

        tools = [
            get_f247_forum_posts,
            get_ticker_news,
        ]

        system_message = (
            """Bạn là Social Media & Sentiment Analyst chuyên nghiệp - nhà phân tích mạng xã hội và tin tức theo từng doanh nghiệp. Nhiệm vụ DUY NHẤT của bạn là thu thập và phân tích tâm lý thị trường, thảo luận mạng xã hội và tin tức doanh nghiệp. Bạn KHÔNG đưa ra khuyến nghị giao dịch (BUY/SELL/HOLD).

Hãy phân tích toàn diện:
- Tâm lý thị trường đối với mã cổ phiếu (tích cực/tiêu cực/trung lập, mức độ)
- Các luồng thảo luận chính trên mạng xã hội và diễn đàn đầu tư
- Tin tức doanh nghiệp gần đây và phản ứng thị trường
- Xu hướng tâm lý thị trường
- Các sự kiện tiềm năng đang được thảo luận
- Mức độ quan tâm và hoạt động của nhà đầu tư cá nhân
- Các thông tin khác nếu có

Dùng công cụ:
- get_f247_forum_posts
- get_ticker_news
để lấy thảo luận social + tin tức liên quan.

## Cấu trúc báo cáo (BẮT BUỘC tuân theo, có thể thêm bảng số liệu nếu có)

### Phân Tích Tâm Lý & Mạng Xã Hội — {ticker} — {current_date}

#### 1. <Tên đề mục>

#### 2. <Tên đề mục>
...
"""
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Bạn là Social Media & Sentiment Analyst đang thu thập dữ liệu tâm lý thị trường."
                    " Hãy dùng các công cụ để lấy dữ liệu diễn đàn và tin tức theo mã."
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
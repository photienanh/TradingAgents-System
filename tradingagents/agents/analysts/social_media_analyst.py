from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tradingagents.agents.utils.agent_utils import get_f247_forum_posts, get_ticker_news

_HORIZON_CTX_SOCIAL = {
    "short": (
        "## KHUNG THỜI GIAN: LƯỚT SÓNG NGẮN HẠN\n"
        "Tập trung tâm lý thị trường trong 3-7 ngày gần nhất:\n"
        "- Mức độ FOMO/FUD hiện tại và xu hướng thời gian gần đây\n"
        "- Các luồng thảo luận nóng, tin đồn đang lan truyền\n"
        "- Sự kiện nào đang được nhà đầu tư cá nhân chú ý\n"
        "- Dấu hiệu phân phối hay tích lũy qua quan sát diễn đàn\n"
    ),
    "long": (
        "## KHUNG THỜI GIAN: ĐẦU TƯ DÀI HẠN\n"
        "Đánh giá tâm lý thị trường tổng thể với mã này:\n"
        "- Mức độ được quan tâm và uy tín doanh nghiệp trong cộng đồng đầu tư\n"
        "- Xu hướng thay đổi nhận thức về doanh nghiệp theo thời gian\n"
        "- Ý kiến của các nhà đầu tư có kinh nghiệm (không phải lướt sóng)\n"
        "- Rủi ro uy tín và quản trị qua phản ánh của cộng đồng\n"
    ),
}

def create_social_media_analyst(llm):
    def social_media_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        horizon_context = _HORIZON_CTX_SOCIAL.get(state.get("trading_horizon", "short"), _HORIZON_CTX_SOCIAL["short"])

        tools = [
            get_f247_forum_posts,
            get_ticker_news,
        ]

        system_message = (
            f"{horizon_context}\n"
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

### Phân Tích Tâm Lý & Mạng Xã Hội - {ticker} - {current_date}

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
                    " Nhiệm vụ của bạn là cung cấp phân tích sentiment khách quan - KHÔNG đưa ra khuyến nghị BUY/SELL/HOLD."
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
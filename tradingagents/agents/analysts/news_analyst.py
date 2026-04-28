from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tradingagents.agents.utils.agent_utils import get_news, get_global_news
from tradingagents.dataflows.config import get_config

_HORIZON_CTX_NEWS = {
    "short": (
        "## KHUNG THỜI GIAN: LƯỚT SÓNG NGẮN HẠN\n"
        "Ưu tiên tin tức trong 3-7 ngày gần nhất. Tập trung vào:\n"
        "- Tin có khả năng tác động giá ngay trong 1-5 phiên tới\n"
        "- Ảnh hưởng ngắn hạn: Kết quả kinh doanh bất ngờ, thông tin thâu tóm/thoái vốn, "
        "sự kiện bất thường của doanh nghiệp\n"
        "- Tin vĩ mô ảnh hưởng tâm lý thị trường ngắn hạn (lãi suất, tỷ giá, "
        "động thái Ngân hàng Nhà nước)\n"
        "Không cần phân tích sâu về chiến lược dài hạn hay kế hoạch 5 năm.\n"
    ),
    "long": (
        "## KHUNG THỜI GIAN: ĐẦU TƯ DÀI HẠN\n"
        "Mở rộng phạm vi tin tức 30-90 ngày. Tập trung vào:\n"
        "- Thay đổi chiến lược kinh doanh, M&A, mở rộng ngành\n"
        "- Chính sách vĩ mô dài hạn ảnh hưởng ngành (quy hoạch, luật mới)\n"
        "- Kết quả kinh doanh xu hướng và định hướng của ban lãnh đạo\n"
        "- Thay đổi cơ cấu cổ đông lớn, quản trị doanh nghiệp\n"
    ),
}

def create_news_analyst(llm):
    def news_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        horizon_context = _HORIZON_CTX_NEWS.get(state.get("trading_horizon", "short"), _HORIZON_CTX_NEWS["short"])

        tools = [
            get_news,
            get_global_news,
        ]

        system_message = (
            f"{horizon_context}\n"
            """Bạn là News Analyst chuyên nghiệp - nhà nghiên cứu phân tích tin tức. Nhiệm vụ DUY NHẤT của bạn là thu thập và phân tích tin tức — cả tin tức doanh nghiệp lẫn tin tức vĩ mô. Bạn KHÔNG đưa ra khuyến nghị giao dịch (BUY/SELL/HOLD).

Hãy phân tích toàn diện:
- Tin tức doanh nghiệp trực tiếp
- Tin tức ngành và đối thủ cạnh tranh
- Tin tức vĩ mô trong nước
- Tin tức quốc tế
- Tác động từng tin đến doanh nghiệp
- Các thông tin khác nếu có

Công cụ sử dụng:
- get_news: cho tin theo công ty/chủ đề
- get_global_news: cho tin vĩ mô tổng quát

## Cấu trúc báo cáo (BẮT BUỘC tuân theo có thể thêm bảng số liệu nếu có, nên liệt kê các tin tức chính để làm rõ nguồn thông tin)

### Phân Tích Tin Tức — {ticker} — {current_date}

#### 1. <Tên đề mục>

#### 2. <Tên đề mục>
...
"""
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
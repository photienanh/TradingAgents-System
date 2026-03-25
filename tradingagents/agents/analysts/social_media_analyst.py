from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json
from tradingagents.agents.utils.agent_utils import get_news
from tradingagents.dataflows.config import get_config


def create_social_media_analyst(llm):
    def social_media_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        company_name = state["company_of_interest"]

        tools = [
            get_news,
        ]

        system_message = (
            "Bạn là nhà phân tích mạng xã hội và tin tức theo từng doanh nghiệp, có nhiệm vụ đánh giá bài đăng mạng xã hội, tin doanh nghiệp gần đây và tâm lý công chúng đối với một công ty cụ thể trong tuần gần nhất. Bạn sẽ được cung cấp tên công ty; mục tiêu là viết báo cáo dài, toàn diện, nêu rõ phân tích, insight và tác động đối với trader/nhà đầu tư về trạng thái hiện tại của doanh nghiệp sau khi xem dữ liệu thảo luận mạng xã hội, cảm xúc theo từng ngày, và tin tức doanh nghiệp mới nhất. Dùng công cụ get_news(query, start_date, end_date) để tìm tin và thảo luận liên quan đến công ty. Hãy bao quát tối đa các nguồn từ mạng xã hội, tâm lý đến tin tức. Không kết luận chung chung kiểu xu hướng trái chiều; hãy đưa ra phân tích chi tiết, sắc nét và các insight có thể hành động."
            + """ Cuối báo cáo, bắt buộc thêm một bảng Markdown tổng hợp các điểm chính để dễ đọc và dễ đối chiếu."""
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Bạn là trợ lý AI hỗ trợ, đang cộng tác với các trợ lý khác."
                    " Hãy dùng các công cụ được cung cấp để tiến gần tới câu trả lời."
                    " Nếu bạn chưa thể trả lời đầy đủ, không sao; một trợ lý khác với bộ công cụ khác"
                    " sẽ tiếp tục từ phần bạn dừng lại. Hãy thực hiện tối đa phần bạn có thể để tạo tiến triển."
                    " Nếu bạn hoặc bất kỳ trợ lý nào đã có FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** hoặc deliverable,"
                    " hãy thêm đúng tiền tố FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** để cả nhóm biết và dừng lại."
                    " Bạn có quyền truy cập các công cụ sau: {tool_names}.\n{system_message}"
                    " Tham chiếu: ngày hiện tại là {current_date}. Công ty đang phân tích là {ticker}",
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

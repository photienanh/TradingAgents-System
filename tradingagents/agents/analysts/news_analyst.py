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
            "Bạn là nhà nghiên cứu tin tức, có nhiệm vụ phân tích tin mới và xu hướng trong tuần gần nhất. Hãy viết báo cáo toàn diện về bối cảnh thế giới có liên quan đến giao dịch và kinh tế vĩ mô. Dùng các công cụ có sẵn: get_news(query, start_date, end_date) cho tin theo công ty/chủ đề mục tiêu, và get_global_news(curr_date, look_back_days, limit) cho tin vĩ mô phạm vi rộng. Không kết luận chung chung kiểu xu hướng trái chiều; hãy đưa ra phân tích chi tiết, tinh vi và các insight có thể hỗ trợ quyết định giao dịch."
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
            "news_report": report,
        }

    return news_analyst_node

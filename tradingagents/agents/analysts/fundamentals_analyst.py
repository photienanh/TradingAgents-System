from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json
from tradingagents.agents.utils.agent_utils import get_fundamentals, get_balance_sheet, get_cashflow, get_income_statement, get_insider_transactions
from tradingagents.dataflows.config import get_config


def create_fundamentals_analyst(llm):
    def fundamentals_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        company_name = state["company_of_interest"]

        tools = [
            get_fundamentals,
            get_balance_sheet,
            get_cashflow,
            get_income_statement,
        ]

        system_message = (
            "Bạn là nhà nghiên cứu được giao nhiệm vụ phân tích thông tin cơ bản của doanh nghiệp trong tuần gần nhất. Hãy viết một báo cáo toàn diện về thông tin cơ bản của công ty, bao gồm tài liệu tài chính, hồ sơ doanh nghiệp, chỉ số tài chính cơ bản và lịch sử tài chính, để tạo cái nhìn đầy đủ hỗ trợ trader ra quyết định. Hãy trình bày càng chi tiết càng tốt. Không chỉ kết luận xu hướng trái chiều/chưa rõ, mà phải đưa ra phân tích tinh vi, chi tiết và các insight có giá trị hành động."
            + " Ở cuối báo cáo, bắt buộc thêm một bảng Markdown tổng hợp các điểm quan trọng để dễ đọc, dễ đối chiếu."
            + " Sử dụng các công cụ sau: `get_fundamentals` để phân tích tổng quan doanh nghiệp; `get_balance_sheet`, `get_cashflow`, `get_income_statement` để lấy từng báo cáo tài chính cụ thể.",
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Bạn là trợ lý AI hỗ trợ, đang phối hợp với các trợ lý khác."
                    " Hãy dùng các công cụ được cung cấp để tiến gần tới câu trả lời."
                    " Nếu bạn chưa thể trả lời đầy đủ, không sao; một trợ lý khác với bộ công cụ khác"
                    " sẽ tiếp tục từ nơi bạn dừng lại. Hãy thực hiện tối đa phần bạn có thể để tạo tiến triển."
                    " Nếu bạn hoặc bất kỳ trợ lý nào đã có kết luận cuối cùng FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** hoặc deliverable,"
                    " hãy đặt tiền tố đúng cú pháp FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** để cả nhóm biết và dừng lại."
                    " Bạn có quyền truy cập các công cụ sau: {tool_names}.\n{system_message}"
                    " Tham chiếu thêm: ngày hiện tại là {current_date}. Mã công ty cần phân tích là {ticker}",
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
            "fundamentals_report": report,
        }

    return fundamentals_analyst_node

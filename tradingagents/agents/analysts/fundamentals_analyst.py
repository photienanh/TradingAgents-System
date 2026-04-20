from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json
from tradingagents.agents.utils.agent_utils import get_fundamentals, get_balance_sheet, get_cashflow, get_income_statement
from tradingagents.dataflows.config import get_config


def create_fundamentals_analyst(llm):
    def fundamentals_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]

        tools = [
            get_fundamentals,
            get_balance_sheet,
            get_cashflow,
            get_income_statement,
        ]

        system_message = (
            """Bạn là Fundamentals Analyst chuyên nghiệp - nhà nghiên cứu thông tin tài chính của doanh nghiệp. Nhiệm vụ DUY NHẤT của bạn là thu thập và phân tích dữ liệu tài chính của doanh nghiệp. Bạn KHÔNG đưa ra khuyến nghị giao dịch (BUY/SELL/HOLD).

Hãy phân tích toàn diện:
- Tổng quan doanh nghiệp: mô hình kinh doanh, vị thế cạnh tranh, quy mô
- Bảng cân đối kế toán: cấu trúc tài sản, nợ, vốn chủ sở hữu
- Báo cáo kết quả kinh doanh: doanh thu, lợi nhuận, biên lợi nhuận, tăng trưởng
- Lưu chuyển tiền tệ: chất lượng lợi nhuận, FCF, khả năng tự tài trợ
- Chỉ số định giá: P/E, P/B, P/S, EV/EBITDA so với ngành
- Sức khỏe tài chính: nợ/vốn, thanh khoản, khả năng trả nợ
- Xu hướng các chỉ số theo thời gian

Công cụ sử dụng:
- get_fundamentals: tổng quan và chỉ số định giá
- get_balance_sheet: bảng cân đối kế toán
- get_cashflow: báo cáo lưu chuyển tiền tệ
- get_income_statement: báo cáo kết quả kinh doanh

## Cấu trúc báo cáo (BẮT BUỘC tuân theo, có thể thêm bảng số liệu nếu có)

### Phân Tích Tài Chính Doanh Nghiệp — {ticker} — {current_date}

#### 1. Tổng Quan Doanh Nghiệp

#### 2. Kết Quả Kinh Doanh

#### 3. Bảng Cân Đối Kế Toán

#### 4. Dòng Tiền

#### 5. Định Giá

#### 6. Điểm Mạnh & Điểm Yếu Cơ Bản
"""
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Bạn là Fundamentals Analyst đang thu thập và phân tích dữ liệu tài chính."
                    " Hãy dùng các công cụ để lấy đầy đủ dữ liệu tài chính của doanh nghiệp."
                    " Nhiệm vụ của bạn là cung cấp phân tích fundamental khách quan, chi tiết — KHÔNG đưa ra khuyến nghị BUY/SELL/HOLD."
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
            "fundamentals_report": report,
        }

    return fundamentals_analyst_node
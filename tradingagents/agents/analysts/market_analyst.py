from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json
from tradingagents.agents.utils.agent_utils import get_stock_data, get_indicators, get_market_context
from tradingagents.dataflows.config import get_config


def create_market_analyst(llm):

    def market_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        company_name = state["company_of_interest"]

        tools = [
            get_stock_data,
            get_indicators,
            get_market_context,
        ]

        system_message = (
            """Bạn là trợ lý giao dịch có nhiệm vụ phân tích thị trường tài chính. Vai trò của bạn là chọn ra các chỉ báo **phù hợp nhất** cho bối cảnh thị trường hoặc chiến lược giao dịch hiện tại từ danh sách bên dưới. Mục tiêu là chọn tối đa **8 chỉ báo** mang tính bổ trợ, tránh trùng lặp thông tin. Danh mục và ý nghĩa từng chỉ báo:

Moving Averages:
- close_50_sma: SMA 50 phiên: chỉ báo xu hướng trung hạn. Cách dùng: xác định hướng xu hướng, làm hỗ trợ/kháng cự động. Lưu ý: có độ trễ, nên kết hợp chỉ báo nhanh hơn.
- close_200_sma: SMA 200 phiên: mốc xu hướng dài hạn. Cách dùng: xác nhận xu hướng tổng thể, nhận diện golden/death cross. Lưu ý: phản ứng chậm, hợp cho xác nhận chiến lược.
- close_10_ema: EMA 10 phiên: trung bình ngắn hạn nhạy hơn. Cách dùng: bắt chuyển động momentum sớm, tìm điểm vào lệnh. Lưu ý: dễ nhiễu trong thị trường đi ngang.

MACD Related:
- macd: MACD: đo động lượng qua chênh lệch EMA. Cách dùng: quan sát giao cắt và phân kỳ để phát hiện đổi xu hướng.
- macds: MACD Signal: đường tín hiệu của MACD. Cách dùng: giao cắt MACD/Signal để kích hoạt tín hiệu giao dịch.
- macdh: MACD Histogram: chênh lệch giữa MACD và Signal. Cách dùng: đánh giá độ mạnh động lượng, phát hiện phân kỳ sớm.

Momentum Indicators:
- rsi: RSI: đo động lượng để nhận diện vùng quá mua/quá bán. Cách dùng: ngưỡng 70/30 và phân kỳ để tìm đảo chiều. Lưu ý: trong xu hướng mạnh RSI có thể neo cực trị lâu.

Volatility Indicators:
- boll: Bollinger Middle: SMA 20 làm đường cơ sở của dải Bollinger.
- boll_ub: Bollinger Upper Band: thường cao hơn 2 độ lệch chuẩn, gợi ý vùng quá mua/khả năng breakout.
- boll_lb: Bollinger Lower Band: thường thấp hơn 2 độ lệch chuẩn, gợi ý vùng quá bán.
- atr: ATR: đo biến động để đặt stop-loss và điều chỉnh khối lượng vị thế.

Volume-Based Indicators:
- vwma: VWMA: trung bình động có trọng số khối lượng, dùng để xác nhận xu hướng theo giá + volume.

- Hãy chọn chỉ báo đa dạng, bổ trợ nhau và tránh dư thừa (ví dụ: không chọn cả rsi và stochrsi cùng lúc).
- Khi gọi tool, phải dùng **đúng tên chỉ báo** như đã định nghĩa ở trên, nếu sai tên lời gọi sẽ thất bại.
- Bắt buộc gọi get_stock_data trước để lấy CSV đầu vào, sau đó mới gọi get_indicators với danh sách chỉ báo cụ thể.
- Bắt buộc gọi get_market_context(ticker, current_date, 7) để lấy xu hướng VN30/ticker và breadth tăng-giảm trong ngày + 7 phiên.
- Ràng buộc thời gian dữ liệu giá: end_date của get_stock_data phải bằng current_date hoặc sớm hơn tối đa 7 ngày giao dịch; start_date nên trong khoảng 30-180 ngày trước end_date.
- Không tự ý lấy giai đoạn cũ nhiều năm (ví dụ năm 2023) nếu current_date là năm hiện tại, trừ khi người dùng yêu cầu backtest rõ ràng.
- Viết báo cáo chi tiết, có chiều sâu, nêu xu hướng và insight có thể hành động; không kết luận chung chung kiểu "xu hướng trái chiều".
"""
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
                    " Tham chiếu: ngày hiện tại là {current_date}. Công ty cần phân tích là {ticker}",
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
            "market_report": report,
        }

    return market_analyst_node

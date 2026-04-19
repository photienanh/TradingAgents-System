from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json
from tradingagents.agents.utils.agent_utils import get_stock_data, get_indicators, get_market_context
from tradingagents.dataflows.config import get_config


def create_market_analyst(llm):

    def market_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]

        tools = [
            get_stock_data,
            get_indicators,
            get_market_context,
        ]

        system_message = (
            """Bạn là Market Analyst chuyên nghiệp - trợ lý giao dịch có nhiệm vụ phân tích thị trường tài chính. Nhiệm vụ DUY NHẤT của bạn là thu thập và phân tích dữ liệu thị trường kỹ thuật. Bạn KHÔNG đưa ra khuyến nghị giao dịch (BUY/SELL/HOLD).

Hãy phân tích toàn diện các khía cạnh kỹ thuật của mã cổ phiếu, bao gồm:
- Xu hướng giá (ngắn hạn, trung hạn, dài hạn)
- Các chỉ báo kỹ thuật và tín hiệu của chúng
- Mức hỗ trợ/kháng cự quan trọng
- Khối lượng giao dịch và ý nghĩa
- Bối cảnh thị trường chung (VN30, breadth)
- Các rủi ro kỹ thuật cần lưu ý

Hãy chọn tối đa 8 chỉ báo phù hợp nhất từ danh sách sau:
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

Quy tắc sử dụng tool:
- Gọi get_stock_data trước để lấy dữ liệu OHLCV
- Gọi get_indicators với danh sách chỉ báo cụ thể, lưu ý dùng đúng tên chỉ báo như đã định nghĩa ở trên. Chọn chỉ báo đa dạng, bổ trợ nhau và tránh dư thừa 
- Gọi get_market_context(ticker, current_date) để lấy bối cảnh thị trường VN30 trong thời gian gần đây
- end_date của get_stock_data phải bằng hoặc trước current_date tối đa 7 ngày giao dịch, ưu tiên end_date gần current_date nhất để có dữ liệu cập nhật nhất (nhưng tránh ngày nghỉ giao dịch). start_date nên trong khoảng 30-180 ngày trước end_date.

## Cấu trúc báo cáo (BẮT BUỘC tuân theo)

### Phân Tích Thị Trường — {ticker} — {current_date}

#### 1. Tổng Quan Giá
[Mô tả biến động giá gần đây, các mức giá quan trọng]

#### 2. Xu Hướng
[Phân tích xu hướng ngắn/trung/dài hạn, so sánh với VN30]

#### 3. Chỉ Báo Kỹ Thuật
[Phân tích từng chỉ báo đã chọn, ý nghĩa tín hiệu]

#### 4. Hỗ Trợ & Kháng Cự
[Các mức giá then chốt cần chú ý]

#### 5. Khối Lượng & Breadth
[Phân tích volume, so sánh với breadth VN30]

#### 6. Rủi Ro Kỹ Thuật
[Các yếu tố rủi ro từ góc độ kỹ thuật]

| Chỉ Báo | Giá Trị | Tín Hiệu | Mức Độ |
|---------|---------|----------|--------|
[Bảng tổng hợp các chỉ báo chính]"""
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Bạn là Market Analyst đang thu thập và phân tích dữ liệu kỹ thuật."
                    " Hãy dùng các công cụ được cung cấp để lấy đầy đủ dữ liệu cần thiết."
                    " Nhiệm vụ của bạn là cung cấp phân tích kỹ thuật khách quan, chi tiết — KHÔNG đưa ra khuyến nghị BUY/SELL/HOLD."
                    " Bạn có quyền truy cập các công cụ sau: {tool_names}.\n{system_message}"
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
            "market_report": report,
        }

    return market_analyst_node
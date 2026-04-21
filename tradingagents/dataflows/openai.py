from openai import OpenAI
from typing import Any
from datetime import datetime, timedelta

from .config import get_config


def _extract_response_text(response: Any) -> str:
    """Extract text safely from OpenAI Responses API payload across output variants."""
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    output = getattr(response, "output", None)
    if not isinstance(output, list):
        return str(response)

    text_chunks: list[str] = []
    for item in output:
        content = getattr(item, "content", None)
        if not isinstance(content, list):
            continue
        for part in content:
            text = getattr(part, "text", None)
            if isinstance(text, str) and text.strip():
                text_chunks.append(text)

    if text_chunks:
        return "\n".join(text_chunks)
    return str(response)


def get_stock_news_openai(query, curr_date, look_back_days=30):
    config = get_config()
    client = OpenAI(base_url=config["backend_url"])

    try:
        end_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    except ValueError:
        return "Định dạng curr_date không hợp lệ. Vui lòng dùng yyyy-mm-dd."

    start_dt = end_dt - timedelta(days=max(look_back_days, 30))
    start_date = start_dt.strftime("%Y-%m-%d")
    end_date = end_dt.strftime("%Y-%m-%d")

    response = client.responses.create(
        model=config["quick_think_llm"],
        input=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            f"Hãy tìm kiếm tin tức và thảo luận mạng xã hội về mã/công ty {query} "
                            f"trong giai đoạn từ {start_date} đến {end_date}, tập trung vào thị trường chứng khoán Việt Nam "
                            f"(HOSE, HNX, UPCOM). Chỉ lấy nội dung được đăng trong đúng khoảng thời gian này. "
                            f"Trả kết quả bằng tiếng Việt, nêu rõ nguồn, thời gian đăng, và tóm tắt tác động tích cực/tiêu cực đến cổ phiếu."
                        ),
                    }
                ],
            }
        ],
        text={"format": {"type": "text"}},
        reasoning={},
        tools=[
            {
                "type": "web_search_preview",
                "user_location": {"type": "approximate"},
                "search_context_size": "low",
            }
        ],
        temperature=1,
        max_output_tokens=4096,
        top_p=1,
        store=True,
    )

    return _extract_response_text(response)


def get_global_news_openai(curr_date, look_back_days=7, limit=10):
    config = get_config()
    client = OpenAI(base_url=config["backend_url"])

    response = client.responses.create(
        model=config["quick_think_llm"],
        input=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            f"Hãy tìm tin vĩ mô toàn cầu và trong nước có thể ảnh hưởng đến thị trường chứng khoán Việt Nam "
                            f"trong giai đoạn từ {look_back_days} ngày trước {curr_date} đến {curr_date}. "
                            f"Chỉ lấy bài đăng trong đúng khoảng thời gian này và giới hạn {limit} bài quan trọng nhất. "
                            f"Trả kết quả bằng tiếng Việt, ưu tiên các chủ đề như lãi suất, tỷ giá USD/VND, giá dầu, chính sách tiền tệ, "
                            f"dòng vốn ngoại và tác động đến các nhóm ngành tại Việt Nam."
                        ),
                    }
                ],
            }
        ],
        text={"format": {"type": "text"}},
        reasoning={},
        tools=[
            {
                "type": "web_search_preview",
                "user_location": {"type": "approximate"},
                "search_context_size": "low",
            }
        ],
        temperature=1,
        max_output_tokens=4096,
        top_p=1,
        store=True,
    )

    return _extract_response_text(response)


def get_fundamentals_openai(ticker :str):
    config = get_config()
    client = OpenAI(base_url=config["backend_url"])

    response = client.responses.create(
        model=config["quick_think_llm"],
        input=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            f"Hãy tổng hợp thông tin cơ bản (fundamental) cho mã {ticker} phù hợp thị trường chứng khoán Việt Nam "
                            f"Chỉ lấy dữ liệu trong đúng giai đoạn gần đây. Trả kết quả bằng tiếng Việt dưới dạng bảng, bao gồm tối thiểu: "
                            f"P/E, P/B, P/S, biên lợi nhuận, tăng trưởng doanh thu/lợi nhuận, dòng tiền, nợ vay, ROE/ROA, "
                            f"và nhận định định giá (rẻ/hợp lý/đắt) theo bối cảnh ngành tại Việt Nam."
                        ),
                    }
                ],
            }
        ],
        text={"format": {"type": "text"}},
        reasoning={},
        tools=[
            {
                "type": "web_search_preview",
                "user_location": {"type": "approximate"},
                "search_context_size": "low",
            }
        ],
        temperature=1,
        max_output_tokens=4096,
        top_p=1,
        store=True,
    )

    return _extract_response_text(response)
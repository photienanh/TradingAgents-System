# TradingAgents/graph/signal_processing.py

from langchain_openai import ChatOpenAI


class SignalProcessor:
    """Processes trading signals to extract actionable decisions."""

    def __init__(self, quick_thinking_llm: ChatOpenAI):
        """Initialize with an LLM for processing."""
        self.quick_thinking_llm = quick_thinking_llm

    def process_signal(self, full_signal: str) -> str:
        """
        Process a full trading signal to extract the core decision.

        Args:
            full_signal: Complete trading signal text

        Returns:
            Extracted decision (BUY, SELL, or HOLD)
        """
        messages = [
            (
                "system",
                "Bạn là trợ lý phân tích quyết định đầu tư từ báo cáo của các chuyên gia. Nhiệm vụ của bạn là trích xuất duy nhất một quyết định giao dịch: SELL, BUY hoặc HOLD. Chỉ trả về đúng một từ SELL, BUY hoặc HOLD; không thêm bất kỳ nội dung nào khác.",
            ),
            ("human", full_signal),
        ]

        content = self.quick_thinking_llm.invoke(messages).content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            normalized_parts = []
            for item in content:
                if isinstance(item, str):
                    normalized_parts.append(item)
                elif isinstance(item, dict):
                    normalized_parts.append(str(item.get("text", "")))
            return "".join(normalized_parts).strip()
        return str(content)

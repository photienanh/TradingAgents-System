# TradingAgents/graph/reflection.py

from typing import Dict, Any
from langchain_openai import ChatOpenAI


class Reflector:
    """Handles reflection on decisions and updating memory."""

    def __init__(self, quick_thinking_llm: ChatOpenAI):
        """Initialize the reflector with an LLM."""
        self.quick_thinking_llm = quick_thinking_llm
        self.reflection_system_prompt = self._get_reflection_prompt()

    def _get_reflection_prompt(self) -> str:
        """Get the system prompt for reflection."""
        return """
Bạn là một chuyên gia phân tích tài chính, có nhiệm vụ rà soát quyết định/phân tích giao dịch và đưa ra phản tư chi tiết theo từng bước.
Mục tiêu là tạo ra các nhận định sâu, chỉ ra điểm cần cải thiện và tuân thủ chặt chẽ các hướng dẫn sau:

1. Lập luận:
    - Với mỗi quyết định giao dịch, xác định quyết định đó đúng hay sai. Quyết định đúng thường giúp tăng lợi nhuận, quyết định sai làm giảm hiệu quả.
    - Phân tích các yếu tố đóng góp vào thành công hoặc sai lầm, bao gồm:
      - Bối cảnh và dữ liệu thị trường.
      - Chỉ báo kỹ thuật.
      - Tín hiệu kỹ thuật.
      - Diễn biến giá.
      - Phân tích dữ liệu thị trường tổng thể.
      - Phân tích tin tức.
      - Phân tích mạng xã hội và tâm lý thị trường.
      - Phân tích dữ liệu cơ bản doanh nghiệp.
    - Nêu mức độ quan trọng của từng yếu tố trong quyết định.

2. Cải thiện:
    - Với các quyết định sai, đề xuất cách điều chỉnh để tối ưu lợi nhuận.
    - Liệt kê hành động cải thiện cụ thể, có thể áp dụng thực tế (ví dụ: đổi quyết định từ HOLD sang BUY ở thời điểm phù hợp).

3. Tổng kết:
    - Tóm tắt bài học rút ra từ cả quyết định đúng và sai.
    - Chỉ ra cách áp dụng các bài học này cho tình huống tương lai có đặc điểm tương tự.

4. Câu tóm lược để lưu nhớ:
    - Trích xuất insight quan trọng thành một câu cô đọng (không quá 1000 token).
    - Câu này phải giữ được bản chất của bài học và lập luận để tiện truy hồi sau này.

Hãy đảm bảo đầu ra chi tiết, chính xác, có thể hành động được. Bạn cũng sẽ nhận được các mô tả khách quan về thị trường (giá, chỉ báo kỹ thuật, tin tức, tâm lý) để làm ngữ cảnh tham chiếu.
"""

    def _extract_current_situation(self, current_state: Dict[str, Any]) -> str:
        """Extract the current market situation from the state."""
        curr_market_report = current_state["market_report"]
        curr_sentiment_report = current_state["sentiment_report"]
        curr_news_report = current_state["news_report"]
        curr_fundamentals_report = current_state["fundamentals_report"]

        return f"{curr_market_report}\n\n{curr_sentiment_report}\n\n{curr_news_report}\n\n{curr_fundamentals_report}"

    def _reflect_on_component(
        self, component_type: str, report: str, situation: str, returns_losses
    ) -> str:
        """Generate reflection for a component."""
        messages = [
            ("system", self.reflection_system_prompt),
            (
                "human",
                f"Vai trò cần phản tư: {component_type}\n\nKết quả lời/lỗ: {returns_losses}\n\nNội dung phân tích/quyết định: {report}\n\nBáo cáo thị trường tham chiếu: {situation}",
            ),
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

    def reflect_bull_researcher(self, current_state, returns_losses, bull_memory):
        """Reflect on bull researcher's analysis and update memory."""
        situation = self._extract_current_situation(current_state)
        bull_debate_history = current_state["investment_debate_state"]["bull_history"]

        result = self._reflect_on_component(
            "BULL", bull_debate_history, situation, returns_losses
        )
        bull_memory.add_situations([(situation, result)])

    def reflect_bear_researcher(self, current_state, returns_losses, bear_memory):
        """Reflect on bear researcher's analysis and update memory."""
        situation = self._extract_current_situation(current_state)
        bear_debate_history = current_state["investment_debate_state"]["bear_history"]

        result = self._reflect_on_component(
            "BEAR", bear_debate_history, situation, returns_losses
        )
        bear_memory.add_situations([(situation, result)])

    def reflect_trader(self, current_state, returns_losses, trader_memory):
        """Reflect on trader's decision and update memory."""
        situation = self._extract_current_situation(current_state)
        trader_decision = current_state["trader_investment_plan"]

        result = self._reflect_on_component(
            "TRADER", trader_decision, situation, returns_losses
        )
        trader_memory.add_situations([(situation, result)])

    def reflect_invest_judge(self, current_state, returns_losses, invest_judge_memory):
        """Reflect on investment judge's decision and update memory."""
        situation = self._extract_current_situation(current_state)
        judge_decision = current_state["investment_debate_state"]["judge_decision"]

        result = self._reflect_on_component(
            "INVEST JUDGE", judge_decision, situation, returns_losses
        )
        invest_judge_memory.add_situations([(situation, result)])

    def reflect_risk_manager(self, current_state, returns_losses, risk_manager_memory):
        """Reflect on risk manager's decision and update memory."""
        situation = self._extract_current_situation(current_state)
        judge_decision = current_state["risk_debate_state"]["judge_decision"]

        result = self._reflect_on_component(
            "RISK JUDGE", judge_decision, situation, returns_losses
        )
        risk_manager_memory.add_situations([(situation, result)])

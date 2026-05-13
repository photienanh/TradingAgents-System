from tradingagents.agents.utils.text_sanitize import sanitize_for_prompt


def create_research_manager(llm):
    def research_manager_node(state) -> dict:
        horizon    = state.get("trading_horizon", "short")
        ticker     = state.get("company_of_interest", "N/A")
        trade_date = state.get("trade_date", "N/A")

        history                 = state["investment_debate_state"].get("history", "")
        investment_debate_state = state["investment_debate_state"]


        if horizon == "short":
            decision_criteria = (
                f"## TIÊU CHUẨN QUYẾT ĐỊNH — LƯỚT SÓNG NGẮN HẠN\n"
                "- **BUY**: Có catalyst và momentum rõ ràng hỗ trợ tăng giá trong 2-5 ngày tới\n"
                "- **SELL**: Rủi ro giảm giá ngắn hạn vượt trội, momentum tiêu cực hoặc tin xấu sắp tác động\n"
                "- **HOLD**: Tín hiệu trái chiều, chưa đủ catalyst để vào lệnh\n\n"

                f"## YÊU CẦU OUTPUT\n"
                f"### Quyết Định Nhóm Nghiên Cứu — {ticker} — {trade_date}\n\n"

                "#### Tóm Tắt Tranh Luận\n"
                "**Bull:** [Luận điểm mạnh nhất về momentum/catalyst ngắn hạn]\n"
                "**Bear:** [Rủi ro/downside cụ thể trong 2-5 ngày tới]\n\n"

                "#### Đánh Giá\n"
                "[2-3 câu phân tích, ưu tiên tín hiệu kỹ thuật và ảnh hưởng ngắn hạn]\n\n"

                "#### Quyết Định: **[BUY / SELL / HOLD]**\n\n"

                "#### Kế Hoạch Hành Động\n"
                "- Lý do chính: [1-2 lý do quyết định]\n"
                "- Điều kiện thoát sớm: [tín hiệu nào khiến đóng vị thế ngay]\n"
                "- Rủi ro ngắn hạn cần theo dõi: [1-2 rủi ro quan trọng nhất]"
            )
        else:
            decision_criteria = (
                f"## TIÊU CHUẨN QUYẾT ĐỊNH — ĐẦU TƯ DÀI HẠN\n"
                "- **BUY**: Có luận điểm tăng trưởng rõ ràng, định giá hợp lý, rủi ro được kiểm soát\n"
                "- **NOT BUY**: Thiếu biên an toàn, luận điểm yếu, hoặc rủi ro vượt tiềm năng\n"
                "Không có lựa chọn thứ ba — nếu không đủ chắc chắn thì kết quả là NOT BUY.\n\n"

                f"## YÊU CẦU OUTPUT\n"
                f"### Quyết Định Nhóm Nghiên Cứu — {ticker} — {trade_date}\n\n"

                "#### Tóm Tắt Tranh Luận\n"
                "**Bull:** [Luận điểm tăng trưởng và lý do định giá hợp lý]\n"
                "**Bear:** [Rủi ro cấu trúc hoặc định giá đang quá cao]\n\n"

                "#### Đánh Giá\n"
                "[2-3 câu, ưu tiên fundamental và định giá dài hạn]\n\n"

                "#### Quyết Định: **[BUY / NOT BUY]**\n\n"

                "#### Luận Điểm Cốt Lõi\n"
                "- Luận điểm chính: [lý do nền tảng]\n"
                "- Điều kiện làm sai luận điểm: [rủi ro có thể phá vỡ luận điểm]\n"
                "- Mốc xem xét lại: [khi nào cần đánh giá lại]"
            )

        prompt = (
            f"Bạn là Research Manager — người tổng hợp kết quả tranh luận và đưa ra quyết định đầu tư.\n\n"

            "## NGUYÊN TẮC TRUNG LẬP (BẮT BUỘC)\n"
            "Chỉ dữ liệu và lập luận quyết định kết quả. "
            "Không có hướng nào là 'mặc định' hay 'an toàn hơn'.\n\n"

            "## QUY TRÌNH ĐÁNH GIÁ\n"
            "1. Xác định luận điểm mạnh nhất của Bull và luận điểm mạnh nhất của Bear\n"
            "2. Đánh giá chất lượng bằng chứng: dữ liệu cụ thể > ý kiến chủ quan\n"
            "3. Xét bài học từ các tình huống tương tự trong quá khứ\n"
            "4. Đưa ra quyết định dựa trên bên có bằng chứng thuyết phục hơn\n\n"

            f"Lịch sử tranh luận:\n{sanitize_for_prompt(history)}\n\n"

            f"{decision_criteria}"
        )

        response = llm.invoke(prompt)

        new_investment_debate_state = {
            "judge_decision":  response.content,
            "history":         investment_debate_state.get("history", ""),
            "bear_history":    investment_debate_state.get("bear_history", ""),
            "bull_history":    investment_debate_state.get("bull_history", ""),
            "current_response": response.content,
            "count":           investment_debate_state["count"],
        }

        return {
            "investment_debate_state": new_investment_debate_state,
            "investment_plan":         response.content,
        }

    return research_manager_node
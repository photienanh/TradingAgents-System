from typing import Any, Dict


def _normalize_side(side: str) -> str:
    value = (side or "").strip().lower()
    if value in {"long", "short", "neutral"}:
        return value
    return "neutral"


def create_alphagpt_analyst():
    """Create an AlphaGPT signal agent that contributes to risk debate context."""

    def alphagpt_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")
        alphagpt_history = risk_debate_state.get("alphagpt_history", "")

        signal: Dict[str, Any] = state.get("alphagpt_signal") or {}
        enabled = bool(signal.get("enabled", False))
        side = _normalize_side(str(signal.get("side", "neutral")))
        score = signal.get("score")
        rank = signal.get("rank")
        source = signal.get("source")

        if enabled:
            signal_line = (
                f"Tín hiệu AlphaGPT: side={side.upper()}, score={score}, rank={rank}, source={source}. "
                "Hãy xem đây là bằng chứng định lượng để tham khảo, không phải lệnh tuyệt đối."
            )
        else:
            signal_line = (
                "Tín hiệu AlphaGPT không khả dụng hoặc đang trung tính với mã này. "
                "Hãy tiếp tục dựa trên phân tích định tính từ các analyst khác."
            )

        argument = f"AlphaGPT Analyst: {signal_line}"

        new_risk_debate_state = {
            "history": history + "\n" + argument,
            "risky_history": risk_debate_state.get("risky_history", ""),
            "safe_history": risk_debate_state.get("safe_history", ""),
            "neutral_history": risk_debate_state.get("neutral_history", ""),
            "alphagpt_history": alphagpt_history + "\n" + argument,
            "latest_speaker": "AlphaGPT",
            "current_risky_response": risk_debate_state.get("current_risky_response", ""),
            "current_safe_response": risk_debate_state.get("current_safe_response", ""),
            "current_neutral_response": risk_debate_state.get("current_neutral_response", ""),
            "current_alphagpt_response": argument,
            "judge_decision": risk_debate_state.get("judge_decision", ""),
            "count": risk_debate_state.get("count", 0),
        }

        return {
            "risk_debate_state": new_risk_debate_state,
            "alphagpt_signal": signal,
        }

    return alphagpt_node

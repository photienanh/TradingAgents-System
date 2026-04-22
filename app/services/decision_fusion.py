"""
app/services/decision_fusion.py
AlphaGPT ↔ TradingAgents decision fusion logic.
"""
from typing import Any, Dict, Optional, Tuple


def extract_decision_label(raw_decision: Any) -> str:
    text = str(raw_decision or "").upper()
    if "BUY"  in text: return "BUY"
    if "SELL" in text: return "SELL"
    return "HOLD"


def fuse_decision_with_alphagpt(
    ta_decision: str,
    alpha_signal: Optional[Dict[str, Any]],
    ic_threshold: float = 0.10,
    long_score_threshold: float = 0.5,
    short_score_threshold: float = -1.0,
) -> Tuple[str, str]:
    decision = ta_decision if ta_decision in {"BUY", "SELL", "HOLD"} else "HOLD"

    if not isinstance(alpha_signal, dict) or not alpha_signal.get("enabled"):
        return decision, "AlphaGPT signal unavailable → TA decision kept"

    side   = (alpha_signal.get("side") or "neutral").strip().lower()
    ic_oos = float(alpha_signal.get("ic_oos") or 0.0)
    score  = float(alpha_signal.get("score") or 0.0)

    if side not in ("long", "short"):
        return decision, f"AlphaGPT neutral (side={side}) → TA decision kept"

    if ic_oos < ic_threshold:
        return decision, f"AlphaGPT weak: ic_oos={ic_oos:.4f} < {ic_threshold} → TA decision kept ({decision})"

    if side == "long":
        if score <= long_score_threshold:
            return decision, f"AlphaGPT long weak: z={score:.3f} <= {long_score_threshold} → TA decision kept ({decision})"
        if decision == "HOLD":  return "BUY",  f"AlphaGPT long (ic_oos={ic_oos:.4f}, z={score:.3f}) upgraded HOLD → BUY"
        if decision == "SELL":  return "HOLD", f"Conflict: TA=SELL vs AlphaGPT long (ic_oos={ic_oos:.4f}) → HOLD"
        return "BUY", f"TA=BUY aligned with AlphaGPT long (ic_oos={ic_oos:.4f}, z={score:.3f})"

    # short
    if score >= short_score_threshold:
        return decision, f"AlphaGPT short weak: z={score:.3f} >= {short_score_threshold} → TA decision kept ({decision})"
    if decision == "HOLD":  return "SELL", f"AlphaGPT short (ic_oos={ic_oos:.4f}, z={score:.3f}) downgraded HOLD → SELL"
    if decision == "BUY":   return "HOLD", f"Conflict: TA=BUY vs AlphaGPT short (ic_oos={ic_oos:.4f}) → HOLD"
    return "SELL", f"TA=SELL aligned with AlphaGPT short (ic_oos={ic_oos:.4f}, z={score:.3f})"
"""AlphaGPT quantitative analyst for TradingAgents (new alpha library model)."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from alpha.daily_runner import compute_ticker_signal, load_alpha_definitions

log = logging.getLogger(__name__)


SYSTEM_PROMPT = """Bạn là Senior Quantitative Analyst cho cổ phiếu Việt Nam.
Viết báo cáo định lượng ngắn gọn, thực dụng, tập trung vào:
- Top 5 alpha được chọn theo ic_oos từ alpha_library.json
- Chỉ dùng các chỉ số: ic_oos, sharpe_oos, return_oos
- Bias giao dịch dựa trên signal hôm nay (signal_today) của mã đang phân tích
- Không nhắc family/flipped hoặc các khái niệm cũ.

Bắt buộc dùng đúng cấu trúc:
## Quant Analyst Report — {ticker} — {trade_date}
### 1. Tổng quan tín hiệu hôm nay
### 2. Chất lượng top 5 alpha (global)
### 3. Diễn giải alpha nổi bật
### 4. Rủi ro và điều kiện vô hiệu
### 5. Kết luận bias định lượng
"""


def _fmt(v: Optional[float], ndigits: int = 4) -> str:
    if v is None:
        return "N/A"
    try:
        return f"{float(v):.{ndigits}f}"
    except Exception:
        return "N/A"


def _build_raw_data(ticker: str, trade_date: str) -> Dict[str, Any]:
    top_alphas = load_alpha_definitions(limit=5)
    signal = compute_ticker_signal(ticker=ticker, top_alphas=top_alphas)

    side = str(signal.get("side", "neutral")).upper()
    action = "BUY" if side == "LONG" else ("SELL" if side == "SHORT" else "HOLD")

    rows: List[Dict[str, Any]] = []
    for i, a in enumerate(top_alphas, start=1):
        rows.append(
            {
                "rank": i,
                "id": a.get("id", ""),
                "description": a.get("description", ""),
                "expression": a.get("expression", ""),
                "ic_oos": a.get("ic_oos"),
                "sharpe_oos": a.get("sharpe_oos"),
                "return_oos": a.get("return_oos"),
            }
        )

    return {
        "ticker": ticker,
        "trade_date": trade_date,
        "action": action,
        "side": signal.get("side", "neutral"),
        "signal_today": signal.get("signal_today"),
        "ic_oos": signal.get("ic_oos"),
        "sharpe_oos": signal.get("sharpe_oos"),
        "return_oos": signal.get("return_oos"),
        "top_alphas": rows,
        "used_alphas": signal.get("used_alphas", []),
        "error": signal.get("error"),
    }


def _build_llm_prompt(raw: Dict[str, Any]) -> str:
    alpha_lines = []
    for a in raw["top_alphas"]:
        alpha_lines.append(
            "\n".join(
                [
                    f"- Rank {a['rank']} | {a['id']}",
                    f"  ic_oos={_fmt(a.get('ic_oos'))}, sharpe_oos={_fmt(a.get('sharpe_oos'), 3)}, return_oos={_fmt(a.get('return_oos'), 3)}",
                    f"  expression: {a.get('expression', '')}",
                    f"  description: {a.get('description', '')}",
                ]
            )
        )

    used_lines = []
    for u in raw.get("used_alphas", []):
        used_lines.append(
            f"- rank {u.get('rank')} | {u.get('id')} | signal_today={_fmt(u.get('signal_today'), 3)} | ic_oos={_fmt(u.get('ic_oos'))}"
        )

    return "\n".join(
        [
            f"Ticker: {raw['ticker']}",
            f"Trade date: {raw['trade_date']}",
            f"Action bias: {raw['action']} (side={raw['side']})",
            f"Signal today: {_fmt(raw.get('signal_today'), 3)}",
            f"Aggregate metrics: ic_oos={_fmt(raw.get('ic_oos'))}, sharpe_oos={_fmt(raw.get('sharpe_oos'), 3)}, return_oos={_fmt(raw.get('return_oos'), 3)}",
            "",
            "Top 5 global alphas (sorted by ic_oos):",
            "\n\n".join(alpha_lines) if alpha_lines else "- none",
            "",
            "Alphas effectively used for this ticker signal today:",
            "\n".join(used_lines) if used_lines else "- none",
            "",
            "Write the report in Vietnamese, practical for Bull/Bear debate and final trader decision.",
        ]
    )


def _fallback_plain_report(raw: Dict[str, Any]) -> str:
    lines = [
        f"## Quant Analyst Report — {raw['ticker']} — {raw['trade_date']}",
        "### 1. Tổng quan tín hiệu hôm nay",
        f"Bias hiện tại: {raw['action']} (side={raw['side']}, signal_today={_fmt(raw.get('signal_today'), 3)}).",
        "### 2. Chất lượng top 5 alpha (global)",
        (
            f"Tổng hợp: ic_oos={_fmt(raw.get('ic_oos'))}, "
            f"sharpe_oos={_fmt(raw.get('sharpe_oos'), 3)}, "
            f"return_oos={_fmt(raw.get('return_oos'), 3)}."
        ),
        "### 3. Diễn giải alpha nổi bật",
    ]

    for a in raw.get("top_alphas", []):
        lines.append(
            (
                f"- Alpha {a['rank']} ({a['id']}): ic_oos={_fmt(a.get('ic_oos'))}, "
                f"sharpe_oos={_fmt(a.get('sharpe_oos'), 3)}, return_oos={_fmt(a.get('return_oos'), 3)}"
            )
        )

    lines.extend(
        [
            "### 4. Rủi ro và điều kiện vô hiệu",
            "- Tín hiệu có thể nhiễu khi thị trường đổi regime mạnh hoặc thanh khoản bất thường.",
            "- Nếu signal_today quay về gần 0, ưu tiên giảm mức conviction.",
            "### 5. Kết luận bias định lượng",
            f"- Khuyến nghị định lượng hiện tại: {raw['action']}.",
        ]
    )

    if raw.get("error"):
        lines.append(f"- Lưu ý dữ liệu: {raw['error']}")

    return "\n".join(lines)


def create_alphagpt_analyst(
    llm,
    alpha_formula_dir: str,
    alpha_values_dir: str,
):
    """
    alpha_formula_dir / alpha_values_dir kept for backward compatibility.
    New flow reads from data/alpha_library.json + data/market_data via alpha.daily_runner.
    """

    _ = (alpha_formula_dir, alpha_values_dir)

    def alphagpt_analyst_node(state: Dict[str, Any]) -> Dict[str, Any]:
        ticker = str(state.get("company_of_interest", "")).upper().strip()
        trade_date = str(state.get("trade_date", ""))

        if not ticker:
            return {
                "messages": [],
                "quant_report": "## Quant Analyst Report\n\nTicker không hợp lệ.",
            }

        raw = _build_raw_data(ticker=ticker, trade_date=trade_date)

        if raw.get("error") and not raw.get("top_alphas"):
            return {
                "messages": [],
                "quant_report": (
                    f"## Quant Analyst Report — {ticker} — {trade_date}\n\n"
                    f"Không thể tạo tín hiệu định lượng: {raw['error']}"
                ),
            }

        try:
            user_prompt = _build_llm_prompt(raw)
            messages = [
                ("system", SYSTEM_PROMPT.format(ticker=ticker, trade_date=trade_date)),
                ("human", user_prompt),
            ]
            response = llm.invoke(messages)
            report = response.content
        except Exception as exc:
            log.warning("[AlphaGPT Analyst] LLM generation failed: %s", exc)
            report = _fallback_plain_report(raw)

        return {"messages": [], "quant_report": report}

    return alphagpt_analyst_node

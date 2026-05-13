import logging
from typing import Any, Dict, List, Optional

from alpha.daily_runner import compute_ticker_signal, load_alpha_definitions

log = logging.getLogger(__name__)


SYSTEM_PROMPT = """Bạn là Senior Quantitative Analyst cho cổ phiếu Việt Nam.
Nhiệm vụ của bạn là viết báo cáo phân tích định lượng chuyên sâu cho một mã cổ phiếu,
dựa trên kết quả alpha signals đã được kiểm chứng trên out-of-sample data.
Phong cách viết:
- Chuyên nghiệp nhưng không khô khan — viết như đang thuyết phục một portfolio manager
- Giải thích ý nghĩa kinh tế, không chỉ liệt kê số liệu
- Khi một chỉ số tốt, giải thích TẠI SAO nó tốt và điều đó có nghĩa gì với mã này
- Khi một chỉ số xấu, giải thích RỦI RO cụ thể và mức độ nghiêm trọng
- Kết nối signal với hành vi thị trường thực tế của mã
- Đầu ra phải hữu ích cho Bull/Bear researcher khi tranh luận

Cấu trúc báo cáo BẮT BUỘC (giữ đúng headers):
## Quant Analyst Report — {ticker} — {trade_date}
### **1. Tổng quan tín hiệu**
### **2. Chất lượng alpha**
### **3. Phân tích từng alpha**
### **4. Đánh giá rủi ro / tích cực**
### **5. Kết luận định lượng**
### **Các alpha được sử dụng:** <Viết lại các công thức alpha (chỉ liệt kê formula bằng gạch đầu dòng, không viết gì thêm)>
    - alpha = ...
    - alpha = ...
    - ...
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

    rows: List[Dict[str, Any]] = []
    for i, a in enumerate(top_alphas, start=1):
        rows.append(
            {
                "rank": i,
                "id": a.get("id", ""),
                "description": a.get("description", ""),
                "hypothesis": a.get("hypothesis", ""),
                "formula": a.get("formula", ""),
                "ic_oos": a.get("ic_oos"),
                "sharpe_oos": a.get("sharpe_oos"),
                "return_oos": a.get("return_oos"),
            }
        )

    return {
        "ticker": ticker,
        "trade_date": trade_date,
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
                    f"  IC={_fmt(a.get('ic_oos'))}, Sharpe={_fmt(a.get('sharpe_oos'), 3)}, Return={_fmt(a.get('return_oos'), 3)}",
                    f"  formula: {a.get('formula', '')}",
                    f"  description: {a.get('description', '')}",
                    f"  hypothesis: {a.get('hypothesis', '')}",
                ]
            )
        )

    used_lines = []
    for u in raw.get("used_alphas", []):
        used_lines.append(
            f"- rank {u.get('rank')} | {u.get('id')} | signal_today={_fmt(u.get('signal_today'), 3)} | IC={_fmt(u.get('ic_oos'))}"
        )

    return "\n".join(
        [
            f"Mã cổ phiếu: {raw['ticker']}",
            f"Ngày giao dịch: {raw['trade_date']}",
            f"Tín hiệu alpha hôm nay: {_fmt(raw.get('signal_today'), 3)} - {raw.get('side', 'neutral').upper()}",
            f"Trung bình chỉ số đánh giá alpha: Avg IC={_fmt(raw.get('ic_oos'))}, Avg Sharpe={_fmt(raw.get('sharpe_oos'), 3)}, Avg Return={_fmt(raw.get('return_oos'), 3)}",
            "",
            "Các alpha được sử dụng (sorted by IC):",
            "\n\n".join(alpha_lines) if alpha_lines else "- none",
            "",
            "Ảnh huởng của các alpha này lên tín hiệu hôm nay:",
            "\n".join(used_lines) if used_lines else "- none",
            "",
        ]
    )


def _fallback_plain_report(raw: Dict[str, Any]) -> str:
    lines = [
        f"## Quant Analyst Report — {raw['ticker']} — {raw['trade_date']}",
        "### 1. Tổng quan tín hiệu hôm nay",
        f"Bias hiện tại: (side={raw['side'].upper()}, signal_today={_fmt(raw.get('signal_today'), 3)}).",
        "### 2. Chất lượng top alpha",
        (
            f"Tổng hợp: Avg IC={_fmt(raw.get('ic_oos'))}, "
            f"Avg Sharpe={_fmt(raw.get('sharpe_oos'), 3)}, "
            f"Avg Return={_fmt(raw.get('return_oos'), 3)}."
        ),
        "### 3. Diễn giải alpha nổi bật",
    ]

    for a in raw.get("top_alphas", []):
        lines.append(
            (
                f"- Alpha {a['rank']} ({a['id']}): IC={_fmt(a.get('ic_oos'))}, "
                f"Sharpe={_fmt(a.get('sharpe_oos'), 3)}, Return={_fmt(a.get('return_oos'), 3)}"
            )
        )

    lines.extend(
        [
            "### 4. Rủi ro và điều kiện vô hiệu",
            "- Tín hiệu có thể nhiễu khi thị trường đổi regime mạnh hoặc thanh khoản bất thường.",
            "- Nếu signal_today quay về gần 0, ưu tiên giảm mức conviction.",
        ]
    )

    if raw.get("error"):
        lines.append(f"- Lưu ý dữ liệu: {raw['error']}")

    return "\n".join(lines)


def create_alpha_analyst(llm):
    """
    alpha_formula_dir / alpha_values_dir kept for backward compatibility.
    New flow reads from data/alpha_library.json + data/market_data via alpha.daily_runner.
    """

    def alpha_analyst_node(state: Dict[str, Any]) -> Dict[str, Any]:
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
            log.warning("[Alpha Analyst] LLM generation failed: %s", exc)
            report = _fallback_plain_report(raw)

        return {"messages": [], "quant_report": report}

    return alpha_analyst_node

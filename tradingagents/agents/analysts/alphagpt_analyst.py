"""
tradingagents/agents/analysts/alphagpt_analyst.py
═══════════════════════════════════════════════════════════════════════
AlphaGPT Quantitative Analyst — Tầng 1 trong TradingAgents pipeline.

Cách hoạt động (two-pass):
  Pass 1: Đọc JSON/CSV → build structured raw data (không LLM)
  Pass 2: LLM nhận raw data → viết report diễn giải sâu

Report được inject vào state["quant_report"] để Bull/Bear Researcher,
Trader, Risk Manager dùng như evidence định lượng.
"""

from __future__ import annotations

import os
import json
import logging
from typing import Optional

import numpy as np
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate

log = logging.getLogger(__name__)

# ─── Ngưỡng tham chiếu (dùng để build context cho LLM) ───────────────

IC_EXCELLENT  = 0.12
IC_GOOD       = 0.06
IC_WEAK       = 0.015
SHARPE_STRONG = 2.0
SHARPE_OK     = 0.8
SIGNAL_STRONG = 1.5
SIGNAL_MEDIUM = 0.8

FAMILY_MEANING = {
    "momentum":          "bắt xu hướng giá đang hình thành, dự đoán tiếp diễn",
    "mean_reversion":    "phát hiện giá đi quá xa mức cân bằng, dự đoán đảo chiều",
    "volume_flow":       "theo dõi dòng tiền thực qua volume/OBV, xác nhận xu hướng",
    "volatility":        "đánh giá regime biến động, nhận diện breakout tiềm năng",
    "sentiment_catalyst":"kết hợp tâm lý thị trường với tín hiệu kỹ thuật",
    "correlation":       "khai thác tương quan giá-volume để phát hiện bất thường",
    "pattern":           "nhận diện mô hình nến và cấu trúc giá lặp lại",
}


# ═══════════════════════════════════════════════════════════════════════
# PASS 1 — Build raw structured data (không LLM)
# ═══════════════════════════════════════════════════════════════════════

def _compute_composite(
    alpha_values_path: str,
    alpha_meta: dict,
) -> tuple[float | None, pd.Series | None]:
    if not os.path.exists(alpha_values_path):
        return None, None
    try:
        av = pd.read_csv(alpha_values_path, index_col="time", parse_dates=True)
        av.index = pd.to_datetime(av.index).normalize()
        av = av.replace([np.inf, -np.inf], np.nan)
    except Exception:
        return None, None

    ok = [a for a in alpha_meta.get("alphas", []) if a.get("status") == "OK"]
    if not ok:
        return None, None

    total = sum(a.get("score", 0.0) for a in ok) or 1.0
    signal = pd.Series(0.0, index=av.index)
    for a in ok:
        col = f"alpha_{a['id']}"
        if col in av.columns:
            signal += (a.get("score", 0.0) / total) * av[col].fillna(0.0)

    mu  = signal.rolling(60, min_periods=20).mean()
    std = signal.rolling(60, min_periods=20).std()
    z   = (signal - mu) / (std + 1e-9)
    if z.empty or z.isna().all():
        return None, None
    latest = float(z.dropna().iloc[-1])
    return (latest if np.isfinite(latest) else None), z


def _build_raw_data(
    ticker: str,
    trade_date: str,
    alpha_meta: dict,
    alpha_values_path: str,
) -> dict:
    """
    Tổng hợp toàn bộ số liệu thô thành một dict có cấu trúc.
    LLM sẽ nhận dict này để diễn giải.
    """
    ok_alphas   = [a for a in alpha_meta.get("alphas", []) if a.get("status") == "OK"]
    total_score = sum(a.get("score", 0.0) for a in ok_alphas) or 1.0
    n_rounds    = alpha_meta.get("n_rounds", 1)
    n_rows      = alpha_meta.get("n_rows", 0)

    latest_z, _ = _compute_composite(alpha_values_path, alpha_meta)

    # Composite signal description
    if latest_z is not None:
        abs_z = abs(latest_z)
        if abs_z >= SIGNAL_STRONG:
            signal_strength = "rất mạnh"
        elif abs_z >= SIGNAL_MEDIUM:
            signal_strength = "vừa"
        elif abs_z > 0.3:
            signal_strength = "yếu"
        else:
            signal_strength = "trung tính"
        signal_direction = "LONG (mua)" if latest_z > 0 else ("SHORT (bán)" if latest_z < 0 else "NEUTRAL")
    else:
        signal_strength  = "không xác định"
        signal_direction = "NEUTRAL"

    # Portfolio quality
    avg_ic_oos = float(np.mean([abs(a.get("ic_oos") or 0) for a in ok_alphas])) if ok_alphas else 0
    avg_sh_oos = float(np.mean([a.get("sharpe_oos") or 0 for a in ok_alphas]))  if ok_alphas else 0

    if avg_ic_oos >= IC_EXCELLENT:
        ic_quality = "rất mạnh (> 0.12) — predictive power cao, hiếm gặp"
    elif avg_ic_oos >= IC_GOOD:
        ic_quality = "tốt (0.06–0.12) — đáng tin cậy cho giao dịch"
    elif avg_ic_oos >= IC_WEAK:
        ic_quality = "yếu nhưng có tín hiệu (0.015–0.06) — dùng thận trọng"
    else:
        ic_quality = f"dưới ngưỡng ({avg_ic_oos:.4f} < {IC_WEAK}) — không đáng tin cậy"

    if avg_sh_oos >= SHARPE_STRONG:
        sh_quality = f"xuất sắc ({avg_sh_oos:.2f} > 2) — risk-adjusted return rất tốt"
    elif avg_sh_oos >= SHARPE_OK:
        sh_quality = f"chấp nhận được ({avg_sh_oos:.2f})"
    elif avg_sh_oos >= 0:
        sh_quality = f"thấp ({avg_sh_oos:.2f}) — lợi nhuận chưa bù đắp rủi ro"
    else:
        sh_quality = f"âm ({avg_sh_oos:.2f}) — cảnh báo nghiêm trọng"

    # Cảnh báo kỹ thuật
    warnings = []
    n_flipped = sum(1 for a in ok_alphas if a.get("flipped"))
    n_gp      = sum(1 for a in ok_alphas if a.get("gp_enhanced"))
    overfit   = [a for a in ok_alphas
                 if a.get("ic") and a.get("ic_oos")
                 and abs(a["ic"]) > 3 * abs(a["ic_oos"]) + 0.01]
    high_to   = [a for a in ok_alphas if (a.get("turnover") or 0) > 1.5]
    weak_ic   = [a for a in ok_alphas if abs(a.get("ic_oos") or 0) < IC_WEAK]

    if n_flipped:
        warnings.append(
            f"{n_flipped}/{len(ok_alphas)} alpha bị flip — tín hiệu gốc đi ngược chiều "
            f"dự đoán, đã đảo chiều tự động. Cẩn thận khi regime thay đổi đột ngột."
        )
    if overfit:
        ids = [str(a["id"]) for a in overfit]
        warnings.append(
            f"Alpha {', '.join(ids)} có IC_IS >> IC_OOS (ratio > 3×) — "
            f"khả năng overfit cao, hiệu suất thực tế thường thấp hơn in-sample nhiều."
        )
    if high_to:
        ids = [str(a["id"]) for a in high_to]
        warnings.append(
            f"Alpha {', '.join(ids)} turnover > 1.5 — chi phí giao dịch thực tế "
            f"(spread + phí) có thể xói mòn đáng kể lợi nhuận lý thuyết."
        )
    if weak_ic:
        ids = [str(a["id"]) for a in weak_ic]
        warnings.append(
            f"Alpha {', '.join(ids)} IC_OOS < {IC_WEAK} — gần như không có "
            f"predictive power, nên loại trong vòng refinement tiếp theo."
        )
    if latest_z is not None and abs(latest_z) > 3.5:
        warnings.append(
            f"Composite z-score = {latest_z:.2f}σ cực đoan — "
            f"có thể là data anomaly hoặc sự kiện bất thường. Kiểm tra lại giá/volume."
        )

    # Chi tiết từng alpha
    alpha_details = []
    for a in sorted(ok_alphas, key=lambda x: x.get("score", 0), reverse=True):
        ic_oos_v  = a.get("ic_oos")
        sh_oos_v  = a.get("sharpe_oos")
        turnover  = a.get("turnover")
        ic_is_v   = a.get("ic")
        weight_pct = round(a.get("score", 0) / total_score * 100)

        # Đánh giá từng chỉ số
        if ic_oos_v is not None:
            v = abs(ic_oos_v)
            if v >= IC_EXCELLENT:
                ic_eval = f"rất mạnh — top tier, hiếm gặp IC_OOS > {IC_EXCELLENT}"
            elif v >= IC_GOOD:
                ic_eval = f"tốt — đáng tin cậy, nằm trong vùng target"
            elif v >= IC_WEAK:
                ic_eval = "biên tế — có tín hiệu nhưng yếu, cần thêm confirmation"
            else:
                ic_eval = "không đáng tin — dưới ngưỡng minimum viable"
        else:
            ic_eval = "N/A"

        if sh_oos_v is not None:
            if sh_oos_v >= SHARPE_STRONG:
                sh_eval = "xuất sắc — lợi nhuận trên rủi ro rất cao"
            elif sh_oos_v >= SHARPE_OK:
                sh_eval = "ổn — risk-adjusted return chấp nhận được"
            elif sh_oos_v >= 0:
                sh_eval = "thấp — lợi nhuận chưa tương xứng rủi ro"
            else:
                sh_eval = "âm — chiến lược đang thua lỗ trên OOS data"
        else:
            sh_eval = "N/A"

        if turnover is not None:
            if turnover > 1.5:
                to_eval = f"cao ({turnover:.2f}) — phải giao dịch thường xuyên, chi phí lớn"
            elif turnover > 0.8:
                to_eval = f"vừa ({turnover:.2f}) — chấp nhận được"
            else:
                to_eval = f"thấp ({turnover:.2f}) — ít giao dịch, chi phí thấp"
        else:
            to_eval = "N/A"

        # IS vs OOS divergence
        if ic_is_v and ic_oos_v:
            ratio = abs(ic_is_v) / (abs(ic_oos_v) + 1e-9)
            if ratio > 3:
                overfit_note = f"OVERFIT WARNING: IC_IS={ic_is_v:.4f} cao hơn IC_OOS {ratio:.1f}× — mô hình học thuộc data"
            elif ratio > 1.5:
                overfit_note = f"IS/OOS divergence nhẹ ({ratio:.1f}×) — bình thường"
            else:
                overfit_note = f"IS/OOS nhất quán ({ratio:.1f}×) — tốt"
        else:
            overfit_note = ""

        alpha_details.append({
            "id":           a["id"],
            "family":       a.get("family", "unknown"),
            "family_desc":  FAMILY_MEANING.get(a.get("family", ""), ""),
            "idea":         a.get("idea", ""),
            "hypothesis":   a.get("hypothesis", ""),
            "expression":   (a.get("expression", "").replace("alpha = ", "")[:120] + "...") if len(a.get("expression","")) > 120 else a.get("expression","").replace("alpha = ",""),
            "weight_pct":   weight_pct,
            "ic_oos":       f"{ic_oos_v:+.4f}" if ic_oos_v is not None else "N/A",
            "ic_oos_eval":  ic_eval,
            "ic_is":        f"{ic_is_v:+.4f}" if ic_is_v is not None else "N/A",
            "ic_5d":        f"{a.get('ic_5d'):+.4f}" if a.get("ic_5d") is not None else "N/A",
            "sharpe_oos":   f"{sh_oos_v:.3f}" if sh_oos_v is not None else "N/A",
            "sharpe_eval":  sh_eval,
            "turnover":     f"{turnover:.3f}" if turnover is not None else "N/A",
            "turnover_eval":to_eval,
            "score":        f"{a.get('score', 0):.4f}",
            "flipped":      a.get("flipped", False),
            "gp_enhanced":  a.get("gp_enhanced", False),
            "overfit_note": overfit_note,
        })

    families = list({a["family"] for a in alpha_details})

    return {
        "ticker":           ticker,
        "trade_date":       trade_date,
        "n_rows":           n_rows,
        "n_rounds":         n_rounds,
        "n_alphas_ok":      len(ok_alphas),
        "n_gp_enhanced":    n_gp,
        "n_flipped":        n_flipped,
        # Composite signal
        "composite_z":      f"{latest_z:.3f}" if latest_z is not None else "N/A",
        "signal_direction": signal_direction,
        "signal_strength":  signal_strength,
        # Portfolio quality
        "avg_ic_oos":       f"{avg_ic_oos:.4f}",
        "ic_quality":       ic_quality,
        "avg_sharpe_oos":   f"{avg_sh_oos:.3f}",
        "sharpe_quality":   sh_quality,
        # Alpha details
        "alpha_details":    alpha_details,
        "families":         families,
        # Cảnh báo
        "warnings":         warnings,
    }


# ═══════════════════════════════════════════════════════════════════════
# PASS 2 — LLM diễn giải
# ═══════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """Bạn là Senior Quantitative Analyst tại một quỹ đầu tư chứng khoán Việt Nam.
Nhiệm vụ của bạn là viết báo cáo phân tích định lượng chuyên sâu cho một mã cổ phiếu,
dựa trên kết quả backtest alpha signals đã được validate trên out-of-sample data.

Phong cách viết:
- Chuyên nghiệp nhưng không khô khan — viết như đang thuyết phục một portfolio manager
- Giải thích ý nghĩa kinh tế, không chỉ liệt kê số liệu
- Khi một chỉ số tốt, giải thích TẠI SAO nó tốt và điều đó có nghĩa gì với mã này
- Khi một chỉ số xấu, giải thích RỦI RO cụ thể và mức độ nghiêm trọng
- Kết nối signal với hành vi thị trường thực tế của mã
- Đầu ra phải hữu ích cho Bull/Bear researcher khi tranh luận

Cấu trúc báo cáo BẮT BUỘC (giữ đúng headers):
## Quant Analyst Report — {ticker} — {trade_date}
### 1. Tổng quan tín hiệu
### 2. Chất lượng mô hình
### 3. Phân tích từng alpha
### 4. Rủi ro và cảnh báo
### 5. Kết luận định lượng"""


def _build_llm_prompt(raw: dict) -> str:
    # Format alpha details thành text dễ đọc cho LLM
    alpha_blocks = []
    for a in raw["alpha_details"]:
        tags = []
        if a["gp_enhanced"]: tags.append("GP-enhanced")
        if a["flipped"]:      tags.append("FLIPPED")
        tag_str = f" [{', '.join(tags)}]" if tags else ""

        block = f"""
  Alpha {a['id']} — {a['family'].upper()}{tag_str} (trọng số {a['weight_pct']}%)
  Ý tưởng: {a['idea']}
  Cơ sở kinh tế: {a['hypothesis']}
  Loại: {a['family']} — {a['family_desc']}
  Formula: {a['expression']}
  
  Số liệu hiệu suất:
  - IC out-of-sample: {a['ic_oos']} → đánh giá: {a['ic_oos_eval']}
  - IC in-sample: {a['ic_is']} | IC horizon 5 ngày: {a['ic_5d']}
  - {a['overfit_note']}
  - Sharpe OOS: {a['sharpe_oos']} → đánh giá: {a['sharpe_eval']}
  - Turnover: {a['turnover']} → đánh giá: {a['turnover_eval']}
  - Score tổng hợp: {a['score']}"""
        alpha_blocks.append(block)

    warnings_text = "\n".join(f"  - {w}" for w in raw["warnings"]) if raw["warnings"] else "  Không có cảnh báo đặc biệt."

    return f"""Dưới đây là dữ liệu backtest AlphaGPT cho mã **{raw['ticker']}** ngày {raw['trade_date']}.
Dữ liệu này được validate trên out-of-sample (30% cuối chuỗi thời gian — không bị overfit).

═══ DỮ LIỆU RAW ═══

TỔNG QUAN:
- Mã: {raw['ticker']} | Ngày: {raw['trade_date']}
- Dữ liệu: {raw['n_rows']} phiên giao dịch | {raw['n_alphas_ok']} alpha validated | {raw['n_rounds']} vòng refinement
- GP-enhanced: {raw['n_gp_enhanced']}/{raw['n_alphas_ok']} | Flipped: {raw['n_flipped']}/{raw['n_alphas_ok']}

COMPOSITE SIGNAL:
- Z-score: {raw['composite_z']}
- Hướng: {raw['signal_direction']}
- Cường độ: {raw['signal_strength']}

CHẤT LƯỢNG PORTFOLIO:
- Avg IC_OOS: {raw['avg_ic_oos']} → {raw['ic_quality']}
- Avg Sharpe_OOS: {raw['avg_sharpe_oos']} → {raw['sharpe_quality']}
- Các alpha families: {', '.join(raw['families'])}

CHI TIẾT TỪNG ALPHA:
{''.join(alpha_blocks)}

CẢNH BÁO KỸ THUẬT:
{warnings_text}

═══ YÊU CẦU ═══

Viết báo cáo phân tích định lượng theo cấu trúc đã quy định.

Cho mỗi alpha, hãy giải thích:
- IC_OOS = {raw['alpha_details'][0]['ic_oos'] if raw['alpha_details'] else 'N/A'} có nghĩa là gì trong thực tế? Tại sao mức này là tốt/xấu/chấp nhận được?
- Sharpe = bao nhiêu thì đủ để trade thực tế với chi phí giao dịch VN30?
- Nếu alpha bị flip, điều đó nói lên gì về thị trường?
- Formula như thế này bắt được signal gì từ price/volume của {raw['ticker']}?

Phần kết luận phải nêu rõ:
- Quant model đang nói gì? (hướng + mức độ tin cậy)
- Bull researcher nên dùng điểm nào để argue?
- Bear researcher nên dùng điểm nào để counter?
"""


# ═══════════════════════════════════════════════════════════════════════
# Agent node factory
# ═══════════════════════════════════════════════════════════════════════

def create_alphagpt_analyst(
    llm,
    alpha_formula_dir: str,
    alpha_values_dir: str,
):
    """
    Factory function tạo AlphaGPT analyst node.

    Args:
        llm:               ChatOpenAI instance (quick_thinking_llm)
        alpha_formula_dir: đường dẫn đến data/alpha_formulas/
        alpha_values_dir:  đường dẫn đến data/alphas/
    """

    def alphagpt_analyst_node(state: dict) -> dict:
        ticker     = state["company_of_interest"]
        trade_date = state["trade_date"]

        meta_path = os.path.join(alpha_formula_dir, f"{ticker}_alphas.json")
        av_path   = os.path.join(alpha_values_dir,  f"{ticker}_alpha_values.csv")

        # ── Trường hợp không có data ──────────────────────────────────
        if not os.path.exists(meta_path):
            log.warning(f"[AlphaGPT Analyst] Không có alpha data cho {ticker}")
            report = (
                f"## Quant Analyst Report — {ticker} — {trade_date}\n\n"
                f"Chưa có dữ liệu AlphaGPT cho mã {ticker}. "
                f"Cần chạy `python -m alpha.pipelines.gen_alpha --ticker {ticker}` "
                f"trước khi phân tích. "
                f"Quyết định phiên này dựa hoàn toàn vào phân tích định tính."
            )
            return {"messages": [], "quant_report": report}

        # ── Pass 1: Build raw data ────────────────────────────────────
        try:
            with open(meta_path, encoding="utf-8") as f:
                alpha_meta = json.load(f)

            raw = _build_raw_data(ticker, trade_date, alpha_meta, av_path)
            ok_count = raw["n_alphas_ok"]
            log.info(f"[AlphaGPT Analyst] Raw data built for {ticker} — {ok_count} alphas")

        except Exception as e:
            log.error(f"[AlphaGPT Analyst] Lỗi đọc data {ticker}: {e}")
            report = (
                f"## Quant Analyst Report — {ticker} — {trade_date}\n\n"
                f"Lỗi khi đọc alpha data: {e}. Bỏ qua quant signal."
            )
            return {"messages": [], "quant_report": report}

        # ── Trường hợp không có alpha nào pass ────────────────────────
        if ok_count == 0:
            report = (
                f"## Quant Analyst Report — {ticker} — {trade_date}\n\n"
                f"AlphaGPT đã chạy {raw['n_rounds']} vòng nhưng không có alpha nào "
                f"pass validation (IC_OOS >= {0.015}, Sharpe_OOS >= {0.2}). "
                f"Quant model không đưa ra khuyến nghị — "
                f"quyết định dựa hoàn toàn vào phân tích định tính."
            )
            return {"messages": [], "quant_report": report}

        # ── Pass 2: LLM diễn giải ─────────────────────────────────────
        try:
            user_prompt = _build_llm_prompt(raw)
            messages = [
                ("system", SYSTEM_PROMPT.format(
                    ticker=ticker, trade_date=trade_date
                )),
                ("human", user_prompt),
            ]
            response = llm.invoke(messages)
            report   = response.content
            log.info(
                f"[AlphaGPT Analyst] LLM report generated for {ticker} "
                f"({len(report)} chars)"
            )

        except Exception as e:
            # Fallback về plain text nếu LLM fail
            log.error(f"[AlphaGPT Analyst] LLM failed cho {ticker}: {e}")
            report = _fallback_plain_report(raw)

        return {"messages": [], "quant_report": report}

    return alphagpt_analyst_node


# ─── Fallback nếu LLM fail ─────────────────────────────────────────────

def _fallback_plain_report(raw: dict) -> str:
    """Plain text report khi LLM không khả dụng."""
    lines = [
        f"## Quant Analyst Report — {raw['ticker']} — {raw['trade_date']}",
        f"[LLM unavailable — raw data summary]\n",
        f"Composite signal: {raw['composite_z']}σ — {raw['signal_direction']} ({raw['signal_strength']})",
        f"Avg IC_OOS: {raw['avg_ic_oos']} ({raw['ic_quality']})",
        f"Avg Sharpe_OOS: {raw['avg_sharpe_oos']} ({raw['sharpe_quality']})",
        "",
    ]
    for a in raw["alpha_details"]:
        lines.append(
            f"Alpha {a['id']} [{a['family']}] weight={a['weight_pct']}% | "
            f"IC_OOS={a['ic_oos']} ({a['ic_oos_eval']}) | "
            f"Sharpe_OOS={a['sharpe_oos']} ({a['sharpe_eval']})"
        )
        lines.append(f"  {a['idea']}")
    if raw["warnings"]:
        lines.append("\nCảnh báo:")
        for w in raw["warnings"]:
            lines.append(f"  - {w}")
    return "\n".join(lines)
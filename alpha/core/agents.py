"""
core/agents.py
══════════════════════════════════════════════════════════════════════
Ba agents theo đúng kiến trúc paper Alpha-GPT (Figure 2 + Figure 5):

  Agent 1 — Trading Idea Polisher
    Nhận: ticker, industry, data context, RAG examples
    Làm: tự tạo trading hypotheses (không cần human input)
    Trả: list structured ideas để Quant Developer implement

  Agent 2 — Quant Developer
    Nhận: structured ideas từ Polisher + operators + data fields
    Làm: implement mỗi idea thành alpha expression Python
    Trả: list alpha definitions (id, family, idea, expression)

  Agent 3 — Analyst
    Nhận: alpha results sau backtest (IC, Sharpe, expression, idea)
    Làm: giải thích tại sao alpha hoạt động/không hoạt động,
         đề xuất hướng cải thiện cho round tiếp theo
    Trả: natural language analysis + refinement directions

Tại sao tách thành 3 agents thay vì 1 prompt duy nhất:
  - Mỗi agent có system prompt riêng → LLM "vào vai" tốt hơn
  - Output của agent trước làm input cho agent sau → chain of thought
  - Analyst feedback được lưu vào history → các rounds sau học từ đó
  - Dễ debug: biết lỗi ở Polisher, Developer hay Analyst

Tất cả agents đều fully autonomous — không cần human input.
"""

import json
import logging
from typing import Optional

from openai import OpenAI

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Shared LLM caller
# ─────────────────────────────────────────────────────────────────────

def _call(
    client: OpenAI,
    model: str,
    system: str,
    user: str,
    temperature: float = 0.7,
    max_tokens: int = 2000,
    json_mode: bool = True,
) -> str:
    """Low-level LLM call. Returns raw text content."""
    kwargs = {}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            log.warning(f"LLM attempt {attempt+1} failed: {e}")
    raise RuntimeError("LLM failed after 3 attempts")


# ─────────────────────────────────────────────────────────────────────
# Agent 1 — Trading Idea Polisher
# ─────────────────────────────────────────────────────────────────────

POLISHER_SYSTEM = """Bạn là Trading Idea Polisher — chuyên gia phân tích thị trường HOSE Việt Nam.

Nhiệm vụ: Dựa vào thông tin về cổ phiếu và thị trường, tự tạo ra các
trading hypotheses chất lượng cao để làm đầu vào cho Quant Developer.

Mỗi hypothesis phải:
1. Mô tả một MARKET INEFFICIENCY cụ thể có thể quan sát được
2. Giải thích MECHANISM tại sao nó tạo ra edge (lợi thế)
3. Chỉ ra DATA SIGNALS nào có thể đo lường được
4. Phù hợp với đặc thù của cổ phiếu và ngành

Đặc thù thị trường HOSE cần lưu ý:
- Retail investors chiếm ~90% → hành vi bầy đàn, overreaction
- Margin trading phổ biến → forced selling khi giảm mạnh
- Thanh khoản không đều → volume spike thường có ý nghĩa
- Tin tức thường phản ánh vào giá ngay trong ngày
- Sector rotation rõ ràng giữa ngân hàng, bất động sản, thép, dầu khí

Phân loại hypothesis theo family:
  momentum        - giá/volume tiếp tục theo hướng hiện tại
  mean_reversion  - giá quay về trung bình sau khi đi quá
  volume_flow     - dòng tiền thông minh vs dòng tiền bầy đàn
  volatility      - thay đổi chế độ biến động giá
  sentiment_catalyst - tin tức/cảm xúc thị trường ảnh hưởng giá
  correlation     - mối quan hệ giữa các chỉ báo
  pattern         - mẫu hình nến, kỹ thuật cổ điển

Output JSON với đúng số hypotheses được yêu cầu."""


def run_polisher(
    client: OpenAI,
    model: str,
    ticker: str,
    industry: str,
    data_context: str,          # data stats, available fields
    sentiment_context: str,     # sentiment quality info
    memory_context: str,        # RAG examples từ alpha memory
    analyst_feedback: str = "", # feedback từ Analyst round trước
    n_ideas: int = 10,
    temperature: float = 0.8,
) -> list[dict]:
    """
    Agent 1: Tự tạo trading hypotheses cho ticker.
    Trả về list of idea dicts.
    """
    feedback_block = ""
    if analyst_feedback:
        feedback_block = f"""
## Analyst Feedback từ Round Trước
{analyst_feedback}

→ Dựa vào feedback trên, tạo hypotheses mới khắc phục các điểm yếu đã chỉ ra.
"""

    user_prompt = f"""
## Ticker: {ticker} | Ngành: {industry}

{data_context}
{sentiment_context}
{memory_context}
{feedback_block}

## Yêu cầu
Tạo ĐÚNG {n_ideas} trading hypotheses cho {ticker}.

Mỗi hypothesis thuộc một FAMILY KHÁC NHAU trong:
[momentum, mean_reversion, volume_flow, volatility,
 sentiment_catalyst, correlation, pattern]

Ưu tiên hypotheses phù hợp với:
- Đặc thù ngành {industry}
- Hành vi retail investors HOSE
- Các market inefficiency thường thấy ở cổ phiếu Việt Nam

Output JSON:
{{
  "ticker": "{ticker}",
  "industry": "{industry}",
  "hypotheses": [
    {{
      "id": 1,
      "family": "<family_name>",
      "title": "<10 từ mô tả ngắn gọn>",
      "condition": "<khi nào market ở trạng thái này?>",
      "mechanism": "<tại sao tạo ra edge?>",
      "prediction": "<dự báo gì về price direction?>",
      "key_signals": ["<signal1>", "<signal2>"],
      "confidence": <1-5>
    }},
    ... × {n_ideas}
  ]
}}
"""

    raw = _call(client, model, POLISHER_SYSTEM, user_prompt,
                temperature=temperature, max_tokens=3000)
    try:
        data = json.loads(raw)
        hypotheses = data.get("hypotheses", [])
        log.info(f"[Polisher] {ticker}: generated {len(hypotheses)} hypotheses")
        return hypotheses
    except Exception as e:
        log.error(f"[Polisher] JSON parse failed: {e}")
        return []


# ─────────────────────────────────────────────────────────────────────
# Agent 2 — Quant Developer
# ─────────────────────────────────────────────────────────────────────

DEVELOPER_SYSTEM = """Bạn là Quant Developer — chuyên gia implement formulaic alpha signals.

Nhiệm vụ: Nhận trading hypotheses từ Trading Idea Polisher và implement
chúng thành Python expressions sử dụng bộ operators đã cho.

Nguyên tắc implementation:
1. Expression PHẢI nhất quán với hypothesis — implement đúng cái idea mô tả
2. Output phải là CONTINUOUS signal (không phải binary 0/1)
3. Combine ≥ 2 data sources từ các nguồn khác nhau
4. Normalize cuối cùng bằng ts_zscore_scale(s,w) hoặc tanh()
5. Tránh signal quá sparse (>65% zeros)

Về sentiment:
- Nếu hypothesis cần sentiment: dùng ts_ema(df['X_S'], 5) để smooth trước
- KHÔNG nhân trực tiếp signal × df['X_S'] → sparse
- Đúng: cwise_mul(signal, add(1.0, cwise_mul(0.2, ts_ema(df['X_S'], 5))))

Syntax bắt buộc:
- Gán kết quả: alpha = <expression>
- Không dùng if/else, for loop, lambda
- ts_zscore_scale(s, w) LUÔN cần 2 args
- div(a, b) thay vì a/b để tránh chia cho 0

Output JSON với đúng số alphas được yêu cầu."""


def run_developer(
    client: OpenAI,
    model: str,
    ticker: str,
    hypotheses: list[dict],
    operators_block: str,
    data_fields_block: str,
    memory_block: str = "",
    temperature: float = 0.6,
) -> list[dict]:
    """
    Agent 2: Implement hypotheses thành alpha expressions.
    Trả về list of alpha definition dicts.
    """
    # Format hypotheses cho developer
    hyp_text = []
    for h in hypotheses:
        hyp_text.append(
            f"[{h['id']}] Family: {h.get('family','?')} | {h.get('title','')}\n"
            f"  Condition: {h.get('condition','')}\n"
            f"  Mechanism: {h.get('mechanism','')}\n"
            f"  Prediction: {h.get('prediction','')}\n"
            f"  Key signals: {', '.join(h.get('key_signals',[]))}"
        )

    user_prompt = f"""
## Implement {len(hypotheses)} Alphas cho {ticker}

### Trading Hypotheses từ Polisher:
{chr(10).join(hyp_text)}

---
{data_fields_block}
{operators_block}
{memory_block}

## Yêu cầu
Implement ĐÚNG {len(hypotheses)} alpha expressions tương ứng với hypotheses trên.
Mỗi expression phải implement ĐÚNG logic của hypothesis — không tự ý đổi hướng.

Output JSON:
{{
  "alphas": [
    {{
      "id": <hypothesis_id>,
      "family": "<family từ hypothesis>",
      "hypothesis": "<1 câu tóm tắt condition + mechanism>",
      "idea": "<1 câu ngắn mô tả signal>",
      "expression": "alpha = <formula>"
    }},
    ... × {len(hypotheses)}
  ]
}}
"""

    raw = _call(client, model, DEVELOPER_SYSTEM, user_prompt,
                temperature=temperature, max_tokens=4000)
    try:
        data = json.loads(raw)
        alphas = data.get("alphas", [])
        log.info(f"[Developer] {ticker}: implemented {len(alphas)} expressions")
        return alphas
    except Exception as e:
        log.error(f"[Developer] JSON parse failed: {e}")
        return []


# ─────────────────────────────────────────────────────────────────────
# Agent 3 — Analyst
# ─────────────────────────────────────────────────────────────────────

ANALYST_SYSTEM = """Bạn là Analyst — chuyên gia phân tích kết quả alpha và đưa ra insights.

Nhiệm vụ: Sau khi Quant Developer tạo alpha và hệ thống backtest,
phân tích kết quả và đưa ra feedback cho round tiếp theo.

Phân tích cần bao gồm:
1. Alpha nào hoạt động tốt và TẠI SAO (liên hệ với hypothesis gốc)
2. Alpha nào yếu và TẠI SAO (hypothesis sai, implementation lỗi, hay data không phù hợp?)
3. Pattern chung: ticker này có thiên hướng mean-reversion hay momentum?
4. Điểm cần cải thiện cụ thể cho round tiếp theo
5. Hypothesis nào nên thử lại với implementation khác

Giữ phân tích THỰC TẾ và ACTIONABLE — không chung chung.
Đặc biệt chú ý đến:
- IC_OOS vs IC_IS chênh lệch lớn → overfit hoặc regime shift
- Sharpe thấp dù IC cao → signal đúng hướng nhưng noisy
- Nhiều alpha bị flip → ticker có hành vi counter-intuitive
- Turnover cao → chi phí giao dịch sẽ ăn vào lợi nhuận

Output JSON với analysis và refinement directions."""


def run_analyst(
    client: OpenAI,
    model: str,
    ticker: str,
    industry: str,
    alpha_results: list[dict],
    round_num: int = 1,
    temperature: float = 0.5,
) -> dict:
    """
    Agent 3: Phân tích kết quả backtest và tạo feedback cho round tiếp.
    Trả về dict chứa analysis text và refinement directions.
    """
    # Format results cho analyst
    results_text = []
    ok_alphas = [r for r in alpha_results if r.get("status") == "OK"]
    err_alphas = [r for r in alpha_results if r.get("status") != "OK"]

    for r in ok_alphas:
        flip_note = " [FLIPPED - signal ngược chiều]" if r.get("flipped") else ""
        gp_note   = " [GP enhanced]" if r.get("gp_enhanced") else ""
        results_text.append(
            f"α{r['id']} [{r.get('family','?')}]{flip_note}{gp_note}\n"
            f"  IC_IS={r.get('ic',0):+.4f}  IC_OOS={r.get('ic_oos',0):+.4f}  "
            f"Sharpe_OOS={r.get('sharpe_oos',0):+.3f}  Turnover={r.get('turnover',0):.3f}\n"
            f"  Idea: {r.get('idea','')}\n"
            f"  Expression: {r.get('expression','')[:100]}"
        )

    for r in err_alphas:
        results_text.append(
            f"α{r['id']} [EVAL_ERROR]: {r.get('error_reason','?')[:80]}\n"
            f"  Expression: {r.get('expression','')[:80]}"
        )

    avg_ic_oos  = sum(abs(r.get("ic_oos")  or 0) for r in ok_alphas) / max(len(ok_alphas), 1)
    avg_sharpe  = sum(r.get("sharpe_oos") or 0  for r in ok_alphas) / max(len(ok_alphas), 1)
    n_flipped   = sum(1 for r in ok_alphas if r.get("flipped"))
    n_gp        = sum(1 for r in ok_alphas if r.get("gp_enhanced"))

    user_prompt = f"""
## Phân Tích Alpha Portfolio — {ticker} ({industry}) | Round {round_num}

### Tổng quan
- Alphas OK: {len(ok_alphas)}/5 | Errors: {len(err_alphas)}
- Avg IC_OOS: {avg_ic_oos:.4f} | Avg Sharpe_OOS: {avg_sharpe:.3f}
- Số alpha bị flip: {n_flipped}/{len(ok_alphas)} (signal ngược chiều hypothesis gốc)
- Số alpha GP enhanced: {n_gp}/{len(ok_alphas)}

### Chi tiết từng Alpha
{chr(10).join(results_text)}

## Yêu cầu phân tích

1. Nhận xét tổng quan về chất lượng portfolio alpha này
2. Phân tích từng alpha: tại sao IC_OOS cao/thấp, signal có ý nghĩa không?
3. Pattern của {ticker}: mean-reversion, momentum, hay mixed?
4. Giải thích tại sao {n_flipped} alpha bị flip (nếu có)
5. Feedback cụ thể cho Polisher để cải thiện hypotheses round tiếp:
   - Hypothesis nào cần thay thế hoàn toàn?
   - Hypothesis nào nên giữ nhưng implement khác?
   - Loại market inefficiency nào nên explore thêm?

Output JSON:
{{
  "overall_assessment": "<2-3 câu đánh giá tổng quan>",
  "ticker_behavior": "<momentum/mean_reversion/mixed và lý do>",
  "alpha_analyses": [
    {{
      "alpha_id": <id>,
      "assessment": "<good/weak/error>",
      "explanation": "<tại sao hoạt động hoặc không>",
      "keep": <true/false>
    }}
  ],
  "flip_explanation": "<giải thích tại sao nhiều alpha bị flip nếu có>",
  "refinement_directions": [
    "<direction 1 cụ thể>",
    "<direction 2 cụ thể>",
    "<direction 3 cụ thể>"
  ],
  "polisher_feedback": "<feedback tổng hợp cho Polisher round tiếp — 3-5 câu>",
  "round_summary": "<1 câu tóm tắt round {round_num}>"
}}
"""

    raw = _call(client, model, ANALYST_SYSTEM, user_prompt,
                temperature=temperature, max_tokens=2500)
    try:
        data = json.loads(raw)
        log.info(f"[Analyst] {ticker} round {round_num}: {data.get('round_summary','')}")
        return data
    except Exception as e:
        log.error(f"[Analyst] JSON parse failed: {e}")
        # Fallback: trả về text analysis dưới dạng simple dict
        return {
            "overall_assessment": raw[:300] if raw else "Analysis unavailable",
            "ticker_behavior": "unknown",
            "alpha_analyses": [],
            "flip_explanation": "",
            "refinement_directions": [],
            "polisher_feedback": "",
            "round_summary": f"Round {round_num} completed",
        }
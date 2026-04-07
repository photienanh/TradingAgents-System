"""
pipelines/gen_alpha.py
Three-agent alpha generation and refinement pipeline.
"""

import os, json, argparse, logging, sys
from pathlib import Path
from copy import deepcopy

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from alpha.core.universe import VN30_SYMBOLS, TICKER_INDUSTRY
from alpha.core.paths import (FEATURES_DIR, SENTIMENT_OUTPUT_DIR,
                        ALPHA_VALUES_DIR, ALPHA_FORMULA_DIR, ALPHA_MEMORY_DIR)
from core import alpha_operators as op
from alpha.core.backtester import (compute_ic, compute_ic_oos, compute_sharpe,
                              compute_sharpe_oos, compute_turnover, composite_score)
from alpha.core.alpha_memory import AlphaMemory, compile_memory_block
from alpha.core.genetic_search import enhance_alpha_population
from alpha.core.agents import run_polisher, run_developer, run_analyst

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── Directories ───────────────────────────────────────────────────────
SENTIMENT_DIR = SENTIMENT_OUTPUT_DIR
for d in [ALPHA_VALUES_DIR, ALPHA_FORMULA_DIR, ALPHA_MEMORY_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────
MAX_ROUNDS             = 5
SEED_OVERSAMPLE        = 10      # Polisher tạo N hypotheses → Developer implement N
GP_ITERATIONS          = 15
GP_ENABLED             = True
IC_THRESHOLD           = 0.015
SHARPE_THRESHOLD       = 0.20
CORR_THRESHOLD         = 0.55
OOS_TEST_RATIO         = 0.30
LLM_MODEL              = "gpt-4o-mini"
SENT_MIN_NONZERO_RATIO = 0.10   # Nâng từ 0.05 lên 0.10 — cần ít nhất 10% ngày có tin tức thực sự

# ── Memory store ──────────────────────────────────────────────────────
memory = AlphaMemory(ALPHA_MEMORY_DIR)


# ═══════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════

def load_data(ticker: str) -> pd.DataFrame:
    feat_path = os.path.join(FEATURES_DIR, f"{ticker}.csv")
    sent_path = os.path.join(SENTIMENT_DIR, f"{ticker}_Full_Sentiment.csv")
    if not os.path.exists(feat_path):
        raise FileNotFoundError(f"Features not found: {feat_path}")
    if not os.path.exists(sent_path):
        raise FileNotFoundError(f"Sentiment not found: {sent_path}")
    df_feat = pd.read_csv(feat_path)
    df_feat["time"] = pd.to_datetime(df_feat["time"]).dt.normalize()
    df_feat = df_feat.set_index("time").sort_index()
    df_sent = pd.read_csv(sent_path, index_col="time", parse_dates=True)
    df_sent.index = pd.to_datetime(df_sent.index).normalize()
    df = df_feat.join(df_sent.sort_index(), how="inner")
    df = df.dropna(subset=["close"]).fillna(0.0)
    log.info(f"[{ticker}] Loaded {len(df)} rows × {len(df.columns)} cols")
    return df


def make_forward_return(df: pd.DataFrame, horizon: int = 1) -> pd.Series:
    return df["close"].pct_change(horizon).shift(-horizon).rename(f"fwd_{horizon}d")


def get_data_stats(df: pd.DataFrame) -> str:
    cols = ["close", "volume", "RSI_14", "MACD", "BB_Upper", "BB_Lower",
            "OBV", "Momentum_3", "Momentum_10"]
    lines = []
    for c in cols:
        if c in df.columns:
            s = df[c].dropna()
            lines.append(
                f"  {c}: μ={s.mean():.2f} σ={s.std():.2f} "
                f"[{s.min():.2f}, {s.max():.2f}]"
            )
    return "\n".join(lines)


def get_sentiment_quality(sent_cols: list[str], df: pd.DataFrame) -> dict:
    result = {}
    for c in sent_cols:
        if c not in df.columns:
            result[c] = {"ok": False, "nonzero": 0.0, "reason": "missing"}
            continue
        s = df[c].dropna()
        if len(s) == 0:
            result[c] = {"ok": False, "nonzero": 0.0, "reason": "empty"}
            continue
        nonzero = float((s != 0).mean())
        ok = nonzero >= SENT_MIN_NONZERO_RATIO
        result[c] = {
            "ok": ok,
            "nonzero": nonzero,
            "reason": "OK" if ok else f"too sparse (nonzero={nonzero:.1%})",
        }
    return result


# ═══════════════════════════════════════════════════════════════════════
# 2. CONTEXT BUILDERS — tạo context strings cho agents
# ═══════════════════════════════════════════════════════════════════════

OPERATOR_SIGNATURES = """
## Operators — syntax STRICT (wrong arg count = runtime error)

### Time-series (series, window)
shift(s,period) | ts_delta(s,period) | ts_delta_ratio(s,period)
ts_mean(s,w) | ts_std(s,w) | ts_sum(s,w) | ts_min(s,w) | ts_max(s,w)
ts_rank(s,w) | ts_median(s,w) | ts_skew(s,w) | ts_kurt(s,w)
ts_zscore_scale(s,w)  ← 2 ARGS bắt buộc
ts_maxmin_scale(s,w)  ← 2 ARGS bắt buộc
ts_corr(s1,s2,w) | ts_cov(s1,s2,w)  ← 3 ARGS
ts_ir(s,w) | ts_linear_reg(s,w) | ts_ema(s,span)
ts_decayed_linear(s,w) | ts_argmaxmin_diff(s,w)
ts_max_diff(s,w) | ts_min_diff(s,w)

### Group-wise
grouped_mean(s,w) | grouped_std(s,w) | grouped_demean(s,w)
grouped_zscore_scale(s,w)

### Element-wise
add(s1,s2) | minus(s1,s2) | cwise_mul(s1,s2) | div(s1,s2)
relu(s) | neg(s) | abso(s) | sign(s) | tanh(s) | log(s) | log1p(s)
pow_op(s,exp) | clip(s,lower,upper)
greater(s1,s2) | less(s1,s2) | cwise_max(s1,s2) | cwise_min(s1,s2)
normed_rank_diff(s1,s2)

### Normalize (no window)
zscore_scale(s) | winsorize_scale(s) | normed_rank(s)
"""


def build_data_context(ticker: str, df: pd.DataFrame,
                       sent_quality: dict) -> str:
    """Context về data fields và statistics — dùng cho Polisher."""
    industry = TICKER_INDUSTRY.get(ticker, "Khác")
    good_sent = [c for c, q in sent_quality.items() if q["ok"]]

    sent_info = ""
    if good_sent:
        by_nonzero = sorted(good_sent,
                            key=lambda c: sent_quality[c]["nonzero"], reverse=True)
        sent_info = f"Sentiment columns (nonzero ≥ {SENT_MIN_NONZERO_RATIO:.0%}):\n"
        for c in by_nonzero[:8]:
            peer = c.replace("_S", "")
            peer_ind = TICKER_INDUSTRY.get(peer, "?")
            sent_info += f"  df['{c}'] ({peer_ind}): nonzero={sent_quality[c]['nonzero']:.1%}\n"
    else:
        sent_info = "Sentiment: all columns too sparse — không dùng sentiment.\n"

    return f"""
## Data Context — {ticker} ({industry})
Số ngày data: {len(df)}

### OHLCV + Technical Indicators
df['open'], df['high'], df['low'], df['close'], df['volume']
df['SMA_5'], df['SMA_20'], df['EMA_10']
df['Momentum_3'], df['Momentum_10']
df['RSI_14'], df['MACD'], df['MACD_Signal']
df['BB_Upper'], df['BB_Middle'], df['BB_Lower'], df['OBV']

### Data Statistics (dùng để calibrate)
{get_data_stats(df)}

### {sent_info}
"""


def build_data_fields_block(sent_quality: dict) -> str:
    """Data fields block cho Quant Developer."""
    good_sent = [c for c, q in sent_quality.items() if q["ok"]]
    sent_list = ", ".join(f"df['{c}']" for c in good_sent[:8]) or "(none)"
    return f"""
## Data Fields (df.index = trading date)
OHLCV:   df['open'], df['high'], df['low'], df['close'], df['volume']
Tech:    df['SMA_5'], df['SMA_20'], df['EMA_10']
         df['Momentum_3'], df['Momentum_10']
         df['RSI_14'], df['MACD'], df['MACD_Signal']
         df['BB_Upper'], df['BB_Middle'], df['BB_Lower'], df['OBV']
Sentiment: {sent_list}
  (encoding: -1=negative, 0=neutral, 1=positive. Mostly 0.)
"""


# ═══════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════

def eval_alpha_expression(expr: str, df: pd.DataFrame):
    namespace = {name: getattr(op, name) for name in dir(op) if not name.startswith("_")}
    namespace.update({"df": df, "np": np, "pd": pd, "op": op})
    exec(expr, namespace)
    result = namespace.get("alpha")
    if not isinstance(result, pd.Series):
        return None
    return result.replace([np.inf, -np.inf], np.nan)


def is_valid_series(series) -> tuple[bool, str]:
    if series is None:
        return False, "None series"
    n = len(series)
    if n == 0:
        return False, "empty"
    n_valid = series.dropna().shape[0]
    if n_valid / n < 0.5:
        return False, f"too many NaN ({n-n_valid}/{n})"
    s = series.dropna()
    if s.std() < 1e-9:
        return False, "constant (std≈0)"
    if (s == 0).all():
        return False, "all zeros"
    zero_ratio = (s == 0).mean()
    if zero_ratio > 0.65:
        return False, f"signal too sparse ({zero_ratio:.0%} zeros)"
    return True, "OK"


def eval_one(alpha_def: dict, df: pd.DataFrame,
             fwd_ret: pd.Series, fwd_ret_5d: pd.Series | None = None) -> dict:
    result = deepcopy(alpha_def)
    result.update({
        "ic": None, "ic_oos": None, "ic_5d": None,
        "sharpe": None, "sharpe_oos": None,
        "turnover": None, "score": 0.0,
        "status": "EVAL_ERROR", "series": None,
        "flipped": False, "gp_enhanced": False,
    })

    try:
        series = eval_alpha_expression(alpha_def["expression"], df)
    except Exception as e:
        result["error_reason"] = str(e)[:120]
        return result

    valid, reason = is_valid_series(series)
    if not valid:
        result["error_reason"] = reason
        return result

    # Expanding normalize — không dùng future data
    exp_mean = series.expanding(min_periods=30).mean()
    exp_std  = series.expanding(min_periods=30).std()
    norm = (series - exp_mean) / (exp_std + 1e-9)
    norm = norm.clip(-5.0, 5.0)

    # Chỉ tính metrics từ ngày signal bắt đầu hợp lệ (bỏ warm-up NaN đầu kỳ).
    eval_mask = norm.notna() & fwd_ret.notna()
    if not eval_mask.any():
        result["error_reason"] = "no valid overlap after warm-up"
        return result
    valid_start = eval_mask[eval_mask].index[0]
    norm_eval = norm.loc[valid_start:]
    fwd_ret_eval = fwd_ret.loc[valid_start:]

    ic_is, ic_oos = compute_ic_oos(norm_eval, fwd_ret_eval, test_ratio=OOS_TEST_RATIO)

    # Auto-flip: dựa vào IC_OOS (honest metric), không phải IC_IS
    # IC_IS có thể dương trong khi IC_OOS âm → không flip → alpha sai hướng ở test
    # Ưu tiên: nếu IC_OOS có giá trị → dùng IC_OOS; fallback sang IC_IS nếu OOS = NaN
    flipped = False
    flip_signal = ic_oos if not np.isnan(ic_oos) else ic_is
    if not np.isnan(flip_signal) and flip_signal < 0:
        norm_eval = -norm_eval
        ic_is  = -ic_is  if not np.isnan(ic_is)  else ic_is
        ic_oos = -ic_oos if not np.isnan(ic_oos) else ic_oos
        flipped = True

    sharpe     = compute_sharpe(norm_eval, fwd_ret_eval)
    sharpe_oos = compute_sharpe_oos(norm_eval, fwd_ret_eval, test_ratio=OOS_TEST_RATIO)
    turnover   = compute_turnover(norm_eval)

    ic_5d = None
    if fwd_ret_5d is not None:
        raw_5d = compute_ic(norm_eval, fwd_ret_5d.loc[valid_start:])
        if not np.isnan(raw_5d):
            ic_5d = round(abs(raw_5d), 6)

    score = composite_score(ic_oos, sharpe_oos, ic_is)

    norm_out = norm.copy()
    norm_out.loc[valid_start:] = norm_eval

    result.update({
        "ic":         round(ic_is, 6)      if not np.isnan(ic_is)      else None,
        "ic_oos":     round(ic_oos, 6)     if not np.isnan(ic_oos)     else None,
        "ic_5d":      ic_5d,
        "sharpe":     round(sharpe, 4)     if not np.isnan(sharpe)     else None,
        "sharpe_oos": round(sharpe_oos, 4) if not np.isnan(sharpe_oos) else None,
        "turnover":   round(turnover, 4)   if not np.isnan(turnover)   else None,
        "score":      score,
        "status":     "OK",
        "series":     norm_out,
        "flipped":    flipped,
    })
    return result


# ═══════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════

def compute_corr_matrix(results: list[dict]) -> pd.DataFrame:
    ok = [r for r in results if r["status"] == "OK" and r.get("series") is not None]
    if not ok:
        return pd.DataFrame()
    df_s = pd.DataFrame({f"alpha_{r['id']}": r["series"] for r in ok})
    return df_s.corr(method="spearman").round(3)


def select_best_alphas(results: list[dict], n: int = 5) -> list[dict]:
    """
    Greedy selection: IC_OOS > 0 bắt buộc, sau đó sort theo score + corr diversity.
    """
    # Chỉ xét alpha có IC_OOS dương (đúng hướng ở test)
    ok = [
        r for r in results
        if r["status"] == "OK" and (r.get("ic_oos") or 0.0) > 0
    ]
    ok.sort(key=lambda r: r.get("score", 0), reverse=True)
    selected = []
    selected_families = set()

    # First pass: 1 alpha per family
    for cand in ok:
        if len(selected) >= n:
            break
        family = cand.get("family", "other")
        if family in selected_families:
            continue
        c_s = cand.get("series")
        if c_s is None:
            continue
        corr_ok = all(
            abs(pd.concat([c_s, e.get("series")], axis=1)
                .dropna().corr(method="spearman").iloc[0, 1]) < CORR_THRESHOLD
            for e in selected if e.get("series") is not None
        )
        if corr_ok:
            selected.append(cand)
            selected_families.add(family)

    # Second pass: fill remaining (any family, still IC_OOS > 0)
    for cand in ok:
        if len(selected) >= n or cand in selected:
            continue
        c_s = cand.get("series")
        if c_s is None:
            continue
        corr_ok = all(
            abs(pd.concat([c_s, e.get("series")], axis=1)
                .dropna().corr(method="spearman").iloc[0, 1]) < CORR_THRESHOLD
            for e in selected if e.get("series") is not None
        )
        if corr_ok:
            selected.append(cand)

    for i, r in enumerate(selected):
        r["id"] = i + 1
    return selected[:n]


def identify_weak_alphas(results: list[dict], corr_matrix: pd.DataFrame,
                         fwd_ret: pd.Series | None = None) -> list[tuple]:
    """
    Đánh dấu alpha yếu cần thay thế.

    Nguyên tắc: CHỈ CẦN MỘT chỉ số xấu là loại ngay — không cần cả hai cùng xấu.

    Hard reject (bất kỳ 1 trong các điều kiện sau):
      1. EVAL_ERROR
      2. IC_OOS <= 0  (dự đoán sai hướng ở test period)
      3. IC_OOS < IC_THRESHOLD (0.015) — quá yếu
      4. Sharpe_OOS < SHARPE_THRESHOLD (0.20) — không profitable
      5. Sharpe_OOS < -0.3 — actively harmful
      6. |corr| >= CORR_THRESHOLD với alpha khác tốt hơn
      7. IC decay > 30%
    """
    from alpha.core.backtester import detect_decay
    weak = []
    weak_ids = set()

    # 1. EVAL_ERROR
    for r in results:
        if r["status"] == "EVAL_ERROR":
            weak.append((r["id"], f"EVAL_ERROR: {r.get('error_reason','?')[:60]}"))
            weak_ids.add(r["id"])

    # 2 & 3. IC_OOS âm hoặc dưới ngưỡng — KHÔNG dùng abs
    for r in results:
        if r["id"] in weak_ids or r["status"] != "OK":
            continue
        ic_oos_raw = r.get("ic_oos")
        if ic_oos_raw is None:
            weak.append((r["id"], "IC_OOS missing"))
            weak_ids.add(r["id"])
            continue
        if ic_oos_raw <= 0:
            weak.append((r["id"],
                f"IC_OOS={ic_oos_raw:.4f} ≤ 0 → sai hướng ở test period"))
            weak_ids.add(r["id"])
        elif ic_oos_raw < IC_THRESHOLD:
            weak.append((r["id"],
                f"IC_OOS={ic_oos_raw:.4f} < threshold {IC_THRESHOLD}"))
            weak_ids.add(r["id"])

    # 4 & 5. Sharpe_OOS thấp
    for r in results:
        if r["id"] in weak_ids or r["status"] != "OK":
            continue
        sh_oos = r.get("sharpe_oos") or 0.0
        if sh_oos < -0.3:
            weak.append((r["id"], f"Sharpe_OOS={sh_oos:.3f} âm → thua lỗ"))
            weak_ids.add(r["id"])
        elif sh_oos < SHARPE_THRESHOLD:
            weak.append((r["id"],
                f"Sharpe_OOS={sh_oos:.3f} < threshold {SHARPE_THRESHOLD}"))
            weak_ids.add(r["id"])

    # 6. High correlation
    if not corr_matrix.empty:
        ok_r = {r["id"]: r for r in results if r["status"] == "OK"}
        ids = list(ok_r.keys())
        for i, id_i in enumerate(ids):
            for id_j in ids[i+1:]:
                ci, cj = f"alpha_{id_i}", f"alpha_{id_j}"
                if ci not in corr_matrix.columns or cj not in corr_matrix.columns:
                    continue
                cv = abs(corr_matrix.loc[ci, cj])
                if cv >= CORR_THRESHOLD:
                    sc_i = ok_r[id_i].get("score", 0)
                    sc_j = ok_r[id_j].get("score", 0)
                    loser = id_i if sc_i <= sc_j else id_j
                    winner = id_j if loser == id_i else id_i
                    if loser not in weak_ids:
                        weak.append((loser, f"|corr|={cv:.3f} with α{winner}"))
                        weak_ids.add(loser)

    # 7. IC decay
    if fwd_ret is not None:
        for r in results:
            if r["id"] in weak_ids or r["status"] != "OK":
                continue
            s = r.get("series")
            if s is None:
                continue
            try:
                d = detect_decay(s, fwd_ret)
                if d.get("decaying"):
                    weak.append((r["id"],
                        f"IC decay −{d.get('drop_pct',0):.0f}%"))
                    weak_ids.add(r["id"])
            except Exception:
                pass

    return weak


# ═══════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════

FALLBACK_ALPHAS = [
    {"id": 99, "family": "momentum",
     "hypothesis": "Price acceleration divergence captures trend.",
     "idea": "Multi-timeframe momentum divergence",
     "expression": "alpha = minus(ts_zscore_scale(df['Momentum_3'],10), ts_zscore_scale(df['Momentum_10'],20))"},
    {"id": 99, "family": "mean_reversion",
     "hypothesis": "RSI extremes signal mean reversion.",
     "idea": "RSI mean-reversion",
     "expression": "alpha = tanh(ts_zscore_scale(minus(50.0, df['RSI_14']),15))"},
    {"id": 99, "family": "volume_flow",
     "hypothesis": "Volume-price divergence reveals distribution.",
     "idea": "Volume-price rank divergence",
     "expression": "alpha = minus(ts_rank(df['volume'],20), ts_rank(ts_delta(df['close'],5),20))"},
    {"id": 99, "family": "volatility",
     "hypothesis": "Bollinger band position predicts reversion.",
     "idea": "BB position signal",
     "expression": "alpha = div(minus(df['close'],df['BB_Middle']), add(minus(df['BB_Upper'],df['BB_Lower']),0.01))"},
    {"id": 99, "family": "correlation",
     "hypothesis": "MACD histogram acceleration captures momentum change.",
     "idea": "MACD histogram delta",
     "expression": "alpha = ts_zscore_scale(ts_delta(minus(df['MACD'],df['MACD_Signal']),3),10)"},
]


# ═══════════════════════════════════════════════════════════════════════
# 6. HELPERS
# ═══════════════════════════════════════════════════════════════════════

def strip_series(results: list[dict]) -> list[dict]:
    return [{k: v for k, v in r.items() if k != "series"} for r in results]


def log_round(ticker: str, rnd: int, results: list[dict], corr: pd.DataFrame) -> None:
    log.info(f"\n{'='*65}\n[{ticker}] ROUND {rnd}\n{'='*65}")
    for r in results:
        if r["status"] == "OK":
            tags = (" [↕]" if r.get("flipped") else "") + (" [GP]" if r.get("gp_enhanced") else "")
            log.info(
                f"  α{r['id']} [{r.get('family','?')}]{tags}\n"
                f"    IC_IS={r.get('ic',0):+.4f}  IC_OOS={r.get('ic_oos',0):+.4f}  "
                f"Sh_OOS={r.get('sharpe_oos',0):+.3f}  Score={r.get('score',0):.4f}\n"
                f"    {r.get('idea','')[:80]}"
            )
        else:
            log.info(f"  α{r['id']} [ERR] {r.get('error_reason','')[:60]}")
    if not corr.empty:
        log.info(f"\n  Corr:\n{corr.to_string()}")


def _rescue_errors(results: list[dict], ticker: str, sent_quality: dict,
                   df: pd.DataFrame, fwd_ret: pd.Series,
                   fwd_ret_5d: pd.Series | None) -> list[dict]:
    """Rescue EVAL_ERROR bằng Developer call với error context."""
    error_results = [r for r in results if r["status"] == "EVAL_ERROR"]
    if not error_results:
        return results

    good_sent = [c for c, q in sent_quality.items() if q["ok"]]
    sent_list = ", ".join(f"df['{c}']" for c in good_sent[:4]) or "(none)"

    rescue_hypotheses = []
    for r in error_results:
        rescue_hypotheses.append({
            "id": r["id"],
            "family": r.get("family", "momentum"),
            "title": "Simple reliable alpha (rescue)",
            "condition": "any market condition",
            "mechanism": "simple technical signal",
            "prediction": "price direction",
            "key_signals": ["close", "volume"],
            "confidence": 2,
            "_error_context": f"Previous expression failed: {r.get('error_reason','?')[:80]}"
        })

    rescue_fields = f"""
## Data Fields
df['close'], df['volume'], df['high'], df['low']
df['RSI_14'], df['MACD'], df['MACD_Signal'], df['BB_Upper'], df['BB_Lower'], df['OBV']
df['SMA_20'], df['EMA_10'], df['Momentum_3'], df['Momentum_10']
Sentiment: {sent_list}
"""

    rescue_ops = OPERATOR_SIGNATURES + """
## RESCUE MODE: Use SIMPLE expressions only (2 operators max)
Safe operators: ts_zscore_scale(s,w), ts_rank(s,w), minus, tanh, ts_delta(s,period)
"""

    log.info(f"[{ticker}] Rescuing {len(error_results)} EVAL_ERROR alphas...")
    new_defs = run_developer(
        client, LLM_MODEL, ticker,
        rescue_hypotheses, rescue_ops, rescue_fields,
        temperature=0.4
    )

    for nd in new_defs:
        nr = eval_one(nd, df, fwd_ret, fwd_ret_5d)
        if nr["status"] == "OK":
            idx = next((i for i, r in enumerate(results) if r["id"] == nr["id"]), None)
            if idx is not None:
                results[idx] = nr
                log.info(f"  [RESCUED] α{nr['id']} IC_OOS={nr.get('ic_oos',0):+.4f}")

    return results


# ═══════════════════════════════════════════════════════════════════════
# 7. MAIN PIPELINE — 3-Agent Flow
# ═══════════════════════════════════════════════════════════════════════

def _run_pipeline(ticker: str, max_rounds: int = MAX_ROUNDS,
                  refine_only: bool = False) -> dict:

    out_json = os.path.join(ALPHA_FORMULA_DIR, f"{ticker}_alphas.json")
    out_csv  = os.path.join(ALPHA_VALUES_DIR,  f"{ticker}_alpha_values.csv")

    df         = load_data(ticker)
    fwd_ret    = make_forward_return(df, horizon=1)
    fwd_ret_5d = make_forward_return(df, horizon=5)
    sent_cols  = [c for c in df.columns if c.endswith("_S")]
    sent_quality = get_sentiment_quality(sent_cols, df)
    industry     = TICKER_INDUSTRY.get(ticker, "Khác")

    good_sent = [c for c, q in sent_quality.items() if q["ok"]]
    log.info(f"[{ticker}] Sentiment: {len(good_sent)}/{len(sent_cols)} usable")

    # RAG memory
    mem_examples = memory.retrieve_diverse(ticker=ticker, top_k=3)
    memory_block = compile_memory_block(mem_examples)
    log.info(f"[{ticker}] Memory: {len(mem_examples)} examples retrieved")

    # Build reusable context strings
    data_context   = build_data_context(ticker, df, sent_quality)
    data_fields    = build_data_fields_block(sent_quality)

    # Sentiment context cho Polisher
    if good_sent:
        sent_context = (
            f"Sentiment columns available: "
            + ", ".join(f"df['{c}']({sent_quality[c]['nonzero']:.0%})" for c in good_sent[:5])
            + "\nDùng làm continuous modifier sau khi smooth bằng ts_ema."
        )
    else:
        sent_context = "Sentiment: all columns too sparse — focus on pure technical."

    history    = []
    results    = []
    analyst_feedback = ""  # Feedback từ Analyst, pass sang Polisher round tiếp

    # ── Refine-only mode: load existing alphas ────────────────────────
    if refine_only and os.path.exists(out_json):
        log.info(f"[{ticker}] Refine-only: loading existing alphas")
        with open(out_json) as f:
            prev = json.load(f)
        existing_defs = [
            {"id": a["id"], "family": a.get("family","?"),
             "hypothesis": a.get("hypothesis",""), "idea": a["idea"],
             "expression": a["expression"]}
            for a in prev.get("alphas", [])
        ]
        results = [eval_one(a, df, fwd_ret, fwd_ret_5d) for a in existing_defs]
        # Lấy analyst feedback từ history nếu có
        if prev.get("history"):
            last = prev["history"][-1]
            if last.get("analyst"):
                analyst_feedback = last["analyst"].get("polisher_feedback", "")
    else:
        # ── Round 1: Full 3-agent generation ─────────────────────────
        log.info(f"\n[{ticker}] === ROUND 1 ===")

        # ── Agent 1: Polisher → hypotheses ───────────────────────────
        log.info(f"[{ticker}] Agent 1: Trading Idea Polisher...")
        hypotheses = run_polisher(
            client, LLM_MODEL,
            ticker=ticker,
            industry=industry,
            data_context=data_context,
            sentiment_context=sent_context,
            memory_context=memory_block,
            analyst_feedback="",   # Round 1: chưa có feedback
            n_ideas=SEED_OVERSAMPLE,
            temperature=0.8,
        )

        if not hypotheses:
            log.warning(f"[{ticker}] Polisher returned empty, using fallbacks")
            results = [eval_one(deepcopy(fb), df, fwd_ret, fwd_ret_5d)
                       for fb in FALLBACK_ALPHAS]
        else:
            # ── Agent 2: Developer → expressions ─────────────────────
            log.info(f"[{ticker}] Agent 2: Quant Developer ({len(hypotheses)} hypotheses)...")
            alpha_defs = run_developer(
                client, LLM_MODEL,
                ticker=ticker,
                hypotheses=hypotheses,
                operators_block=OPERATOR_SIGNATURES,
                data_fields_block=data_fields,
                memory_block=memory_block,
                temperature=0.6,
            )

            if not alpha_defs:
                log.warning(f"[{ticker}] Developer returned empty, using fallbacks")
                alpha_defs = [deepcopy(fb) for fb in FALLBACK_ALPHAS]

            # ── Backtest ──────────────────────────────────────────────
            all_results = [eval_one(a, df, fwd_ret, fwd_ret_5d) for a in alpha_defs]

            for r in all_results:
                tag = "OK" if r["status"] == "OK" else "ERR"
                extra = (f"IC_OOS={r.get('ic_oos',0):+.4f} Sc={r.get('score',0):.4f}"
                         if r["status"] == "OK" else r.get("error_reason","?")[:50])
                log.info(f"  [{tag}] α{r.get('id','?')} [{r.get('family','?')}] {extra}")

            # ── GP Enhancement ────────────────────────────────────────
            if GP_ENABLED:
                log.info(f"[{ticker}] GP enhancement ({len(all_results)} alphas)...")
                all_results = enhance_alpha_population(
                    all_results, df, fwd_ret, fwd_ret_5d,
                    eval_fn=eval_one, n_iterations=GP_ITERATIONS
                )

            results = select_best_alphas(all_results, n=5)

    # Supplement if < 5
    for attempt in range(4):
        if len(results) >= 5:
            break
        n_need = 5 - len(results)
        existing_families = [r.get("family","?") for r in results if r["status"] == "OK"]
        log.warning(f"[{ticker}] Only {len(results)}/5 OK, supplementing {n_need}...")

        extra_hyp = run_polisher(
            client, LLM_MODEL,
            ticker=ticker, industry=industry,
            data_context=data_context,
            sentiment_context=sent_context,
            memory_context=memory_block,
            analyst_feedback=f"Already have families: {existing_families}. "
                             f"Generate {n_need} ideas from OTHER families.",
            n_ideas=n_need,
            temperature=0.9,
        )
        if extra_hyp:
            extra_defs = run_developer(
                client, LLM_MODEL, ticker,
                extra_hyp, OPERATOR_SIGNATURES, data_fields,
                temperature=0.7,
            )
            for ed in extra_defs:
                ed["id"] = len(results) + 1
                er = eval_one(ed, df, fwd_ret, fwd_ret_5d)
                if er["status"] == "OK":
                    results.append(er)
                    log.info(f"  [SUPP] α{er['id']} IC_OOS={er.get('ic_oos',0):+.4f}")
                if len(results) >= 5:
                    break

    # Fallback kỹ thuật — LUÔN đảm bảo đủ 5 alpha
    # Nếu vẫn thiếu sau supplement: dùng fallback không quan tâm family trùng
    if len(results) < 5:
        log.warning(f"[{ticker}] Still {len(results)}/5 — using fallback alphas (no family filter)")
        existing_exprs = {r.get("expression","") for r in results}
        for fb in FALLBACK_ALPHAS:
            if len(results) >= 5:
                break
            # Bỏ filter family — ưu tiên đủ số lượng
            if fb["expression"] in existing_exprs:
                continue
            fb_copy = deepcopy(fb)
            fb_copy["id"] = len(results) + 1
            fb_r = eval_one(fb_copy, df, fwd_ret, fwd_ret_5d)
            if fb_r["status"] == "OK":
                results.append(fb_r)
                existing_exprs.add(fb["expression"])
                log.info(f"  [FALLBACK] α{fb_r['id']} family={fb['family']} IC_OOS={fb_r.get('ic_oos',0):+.4f}")

    # Cực chẳng đã: nếu vẫn < 5, chấp nhận fallback dù đã eval error bằng cách tạo thêm
    # Đây là safety net cuối cùng, không nên xảy ra thường xuyên
    if len(results) < 5:
        log.warning(f"[{ticker}] Emergency: adding minimal technical alphas to reach 5")
        emergency_alphas = [
            {"id": 99, "family": "momentum",
             "hypothesis": "Short-term price momentum.",
             "idea": "Short-term return momentum",
             "expression": "alpha = ts_zscore_scale(ts_delta(df['close'],3),10)"},
            {"id": 99, "family": "volume_flow",
             "hypothesis": "OBV trend signal.",
             "idea": "OBV momentum",
             "expression": "alpha = ts_zscore_scale(ts_delta(df['OBV'],5),20)"},
            {"id": 99, "family": "mean_reversion",
             "hypothesis": "Price deviation from SMA.",
             "idea": "Price vs SMA_5 deviation",
             "expression": "alpha = ts_zscore_scale(minus(df['close'],df['SMA_5']),10)"},
        ]
        for ea in emergency_alphas:
            if len(results) >= 5:
                break
            ea_copy = deepcopy(ea)
            ea_copy["id"] = len(results) + 1
            ea_r = eval_one(ea_copy, df, fwd_ret, fwd_ret_5d)
            if ea_r["status"] == "OK":
                results.append(ea_r)
                log.info(f"  [EMERGENCY] α{ea_r['id']} IC_OOS={ea_r.get('ic_oos',0):+.4f}")

    corr = compute_corr_matrix(results)
    log_round(ticker, 1, results, corr)

    # ── Agent 3: Analyst — phân tích round 1 ─────────────────────────
    log.info(f"[{ticker}] Agent 3: Analyst reviewing round 1...")
    analyst_output = run_analyst(
        client, LLM_MODEL,
        ticker=ticker,
        industry=industry,
        alpha_results=results,
        round_num=1,
    )
    analyst_feedback = analyst_output.get("polisher_feedback", "")
    log.info(f"[{ticker}] Analyst: {analyst_output.get('round_summary','')}")
    log.info(f"[{ticker}] Ticker behavior: {analyst_output.get('ticker_behavior','')}")

    history.append({
        "round": 1,
        "alphas": strip_series(results),
        "analyst": analyst_output,
    })

    # ── Refinement rounds 2-N ─────────────────────────────────────────
    for rnd in range(2, max_rounds + 1):
        weak = identify_weak_alphas(results, corr, fwd_ret=fwd_ret)
        if not weak:
            log.info(f"[{ticker}] All alphas OK — stop at round {rnd - 1}")
            break

        log.info(f"\n[{ticker}] === ROUND {rnd}: Refinement ({len(weak)} weak) ===")
        for aid, reason in weak:
            log.info(f"  → Replace α{aid}: {reason[:70]}")

        weak_ids    = {w[0] for w in weak}
        keep_ids    = {r["id"] for r in results if r["id"] not in weak_ids}
        keep_families = [r.get("family","?") for r in results
                         if r["id"] in keep_ids and r["status"] == "OK"]

        # Analyst feedback + weak context → Polisher
        weak_context = "\n".join(
            f"  α{aid}: {reason}" for aid, reason in weak
        )
        refine_feedback = (
            f"Analyst feedback từ round trước:\n{analyst_feedback}\n\n"
            f"Alphas cần thay thế:\n{weak_context}\n\n"
            f"Families hiện có: {keep_families}\n"
            f"Đề xuất hypotheses thuộc families khác với: {keep_families}"
        )

        # RAG: retrieve relevant examples
        query = " ".join(r.get("idea","") for r in results
                         if r.get("id") in weak_ids)
        refine_mem = memory.retrieve(query, ticker=ticker, top_k=2)
        refine_memory_block = compile_memory_block(refine_mem, "Refinement Examples")

        # Agent 1: Polisher với feedback
        new_hypotheses = run_polisher(
            client, LLM_MODEL,
            ticker=ticker, industry=industry,
            data_context=data_context,
            sentiment_context=sent_context,
            memory_context=refine_memory_block,
            analyst_feedback=refine_feedback,
            n_ideas=len(weak),
            temperature=min(1.0, 0.8 + 0.05 * (rnd - 2)),
        )

        # Agent 2: Developer
        if new_hypotheses:
            new_defs = run_developer(
                client, LLM_MODEL, ticker,
                new_hypotheses, OPERATOR_SIGNATURES, data_fields,
                memory_block=refine_memory_block,
                temperature=0.65,
            )
        else:
            new_defs = []

        # Assign IDs matching what we want to replace
        for i, nd in enumerate(new_defs):
            if i < len(weak):
                nd["id"] = weak[i][0]

        # Backtest + GP
        new_evals = [eval_one(nd, df, fwd_ret, fwd_ret_5d) for nd in new_defs]
        if GP_ENABLED and new_evals:
            new_evals = enhance_alpha_population(
                new_evals, df, fwd_ret, fwd_ret_5d,
                eval_fn=eval_one, n_iterations=10
            )

        # - Bắt buộc: new IC_OOS > 0 (không nhận alpha sai hướng)
        # - Thay nếu: new_score > old_score HOẶC old alpha là hard_weak
        # hard_weak = IC_OOS âm hoặc Sharpe âm → luôn ưu tiên thay dù new_score thấp hơn
        replaced = 0
        hard_weak_ids = {
            w[0] for w in weak
            if "sai hướng" in w[1] or "âm" in w[1] or "≤ 0" in w[1]
        }
        for nr in new_evals:
            tid = nr["id"]
            old_r = next((r for r in results if r["id"] == tid), None)
            if old_r is None:
                continue
            if nr.get("status") != "OK":
                continue

            # Bắt buộc: new alpha phải có IC_OOS dương
            new_ic_oos = nr.get("ic_oos") or 0.0
            if new_ic_oos <= 0:
                log.info(f"  [REJECT] α{tid}: new IC_OOS={new_ic_oos:.4f} ≤ 0")
                continue

            old_score = old_r.get("score", 0.0) if old_r.get("status") == "OK" else 0.0
            new_score = nr.get("score", 0.0)

            # Check corr against kept alphas
            nr_s = nr.get("series")
            corr_ok = True
            if nr_s is not None:
                for kid in keep_ids:
                    kept = next((r for r in results if r["id"] == kid), None)
                    if kept is None or kept.get("series") is None:
                        continue
                    m = pd.concat([nr_s, kept["series"]], axis=1).dropna()
                    if len(m) >= 10:
                        cv = abs(m.iloc[:,0].corr(m.iloc[:,1], method="spearman"))
                        if cv >= CORR_THRESHOLD:
                            corr_ok = False
                            break

            if not corr_ok:
                log.info(f"  [REJECT] α{tid}: high corr with kept alpha")
                continue

            # Replace nếu: score tốt hơn HOẶC old là hard_weak (IC âm / Sharpe âm)
            should_replace = new_score > old_score
            if not should_replace and tid in hard_weak_ids:
                # Old alpha sai hướng → chấp nhận new alpha kém hơn một chút
                should_replace = new_score >= old_score * 0.7 or new_score > 0
                if should_replace:
                    log.info(f"  [FORCE_REPLACE] α{tid}: old was hard_weak")

            if should_replace:
                idx = next(i for i, r in enumerate(results) if r["id"] == tid)
                results[idx] = nr
                keep_ids.add(tid)
                replaced += 1
                log.info(f"  [REPLACED] α{tid}: score {old_score:.4f} → {new_score:.4f} | IC_OOS {old_r.get('ic_oos',0):+.4f} → {new_ic_oos:+.4f}")

        corr = compute_corr_matrix(results)
        log_round(ticker, rnd, results, corr)

        # Agent 3: Analyst
        log.info(f"[{ticker}] Agent 3: Analyst reviewing round {rnd}...")
        analyst_output = run_analyst(
            client, LLM_MODEL,
            ticker=ticker, industry=industry,
            alpha_results=results,
            round_num=rnd,
        )
        analyst_feedback = analyst_output.get("polisher_feedback", "")
        log.info(f"[{ticker}] Analyst: {analyst_output.get('round_summary','')}")

        history.append({
            "round": rnd,
            "replaced": replaced,
            "alphas": strip_series(results),
            "analyst": analyst_output,
        })

        if replaced == 0:
            log.info(f"[{ticker}] No improvement in round {rnd}, continuing...")

    # ── Rescue EVAL_ERRORs ────────────────────────────────────────────
    results = _rescue_errors(results, ticker, sent_quality, df, fwd_ret, fwd_ret_5d)

    # ── Save to memory ────────────────────────────────────────────────
    ok_count = 0
    for r in results:
        if r["status"] == "OK":
            memory.store(ticker, r)
            ok_count += 1
    log.info(f"[{ticker}] Stored {ok_count} alphas to memory")

    # ── Save alpha values CSV ─────────────────────────────────────────
    series_dict = {}
    for r in results:
        col = f"alpha_{r['id']}"
        series_dict[col] = (
            r["series"] if r["status"] == "OK" and r.get("series") is not None
            else pd.Series(0.0, index=df.index)
        )
    df_vals = pd.DataFrame(series_dict, index=df.index)
    df_vals.index.name = "time"
    df_vals.to_csv(out_csv)

    # ── Save formula JSON ─────────────────────────────────────────────
    final = strip_series(results)
    output = {
        "ticker":   ticker,
        "n_rows":   len(df),
        "n_rounds": len(history),
        "alphas":   final,
        "history":  history,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # ── Final log ─────────────────────────────────────────────────────
    ok = [r for r in final if r["status"] == "OK"]
    log.info(f"\n{'='*65}\n[{ticker}] FINAL: {len(ok)}/5 OK | {len(history)} rounds\n{'='*65}")
    for r in ok:
        tags = (" [↕]" if r.get("flipped") else "") + (" [GP]" if r.get("gp_enhanced") else "")
        log.info(
            f"  α{r['id']} [{r.get('family','?')}]{tags}  "
            f"IC_IS={r.get('ic',0):+.4f}  IC_OOS={r.get('ic_oos',0):+.4f}  "
            f"Sh_OOS={r.get('sharpe_oos',0):+.3f}\n    {r.get('idea','')}"
        )

    return output


# ── Public API ────────────────────────────────────────────────────────

def run_single(ticker: str, max_rounds: int = MAX_ROUNDS,
               force: bool = False) -> dict:
    out_json = os.path.join(ALPHA_FORMULA_DIR, f"{ticker}_alphas.json")
    if os.path.exists(out_json) and not force:
        log.info(f"[{ticker}] Skip (exists). Use --force to regenerate.")
        return json.load(open(out_json))
    return _run_pipeline(ticker, max_rounds=max_rounds, refine_only=False)


def run_single_refine(ticker: str, max_rounds: int = MAX_ROUNDS) -> dict:
    out_json = os.path.join(ALPHA_FORMULA_DIR, f"{ticker}_alphas.json")
    if not os.path.exists(out_json):
        log.warning(f"[{ticker}] No existing file — full generation.")
        return _run_pipeline(ticker, max_rounds=max_rounds, refine_only=False)
    return _run_pipeline(ticker, max_rounds=max_rounds, refine_only=True)


def run_all(max_rounds: int = MAX_ROUNDS, force: bool = False) -> None:
    for ticker in VN30_SYMBOLS:
        try:
            out = run_single(ticker, max_rounds=max_rounds, force=force)
            ok  = [a for a in out["alphas"] if a["status"] == "OK"]
            avg = (sum(abs(a.get("ic_oos") or a.get("ic") or 0) for a in ok)
                   / max(len(ok), 1))
            log.info(f"  {ticker}: {len(ok)}/5 OK  avg_IC_OOS={avg:.4f}")
        except Exception as e:
            log.error(f"[{ticker}] FAILED: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Alpha-GPT v4 — 3-Agent Pipeline")
    parser.add_argument("--ticker",      type=str)
    parser.add_argument("--all",         action="store_true")
    parser.add_argument("--rounds",      type=int, default=MAX_ROUNDS)
    parser.add_argument("--force",       action="store_true")
    parser.add_argument("--refine-only", action="store_true")
    parser.add_argument("--no-gp",       action="store_true")
    args = parser.parse_args()

    if args.no_gp:
        GP_ENABLED = False

    if args.ticker:
        t = args.ticker.upper()
        if args.refine_only:
            run_single_refine(t, max_rounds=args.rounds)
        else:
            run_single(t, max_rounds=args.rounds, force=args.force)
    elif args.all:
        run_all(max_rounds=args.rounds, force=args.force)
    else:
        parser.print_help()
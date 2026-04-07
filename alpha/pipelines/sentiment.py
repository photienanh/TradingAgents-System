"""
alpha/pipelines/sentiment.py
Pipeline sentiment: crawl CafeF → score GPT-4o-mini → merge → output.
"""
import os
import sys
import re
import json
import time
import logging
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime
from dateutil.relativedelta import relativedelta
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from openai import OpenAI

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from alpha.core.universe import VN30_SYMBOLS
from alpha.core.paths import RAW_NEWS_DIR, DAILY_SCORES_DIR, SENTIMENT_OUTPUT_DIR, FEATURES_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ===================================================================
# 1. CẤU HÌNH
# ===================================================================
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

RAW_DIR    = RAW_NEWS_DIR
SCORED_DIR = DAILY_SCORES_DIR
OUTPUT_DIR = SENTIMENT_OUTPUT_DIR
for d in [RAW_DIR, SCORED_DIR, OUTPUT_DIR]:
    os.makedirs(d, exist_ok=True)

CAFEF_AJAX_URL = "https://cafef.vn/du-lieu/Ajax/Events_RelatedNews_New.aspx"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

# Sentiment encoding: negative=-1, neutral=0, positive=1
SENTIMENT_MAP = {0: -1, 1: 1, 2: 0}

VN30_RELATED_MAP = {
    "ACB": ["MBB", "TCB", "VPB", "STB", "VCB", "CTG", "BID", "SSI", "VND", "VIB"],
    "BID": ["VCB", "CTG", "MBB", "TCB", "VPB", "ACB", "STB", "HPG", "VIC", "VHM"],
    "CTG": ["BID", "VCB", "MBB", "TCB", "VPB", "ACB", "STB", "HPG", "VIC", "VHM"],
    "DGC": ["CSV", "LAS", "DDV", "BFC", "DPM", "DCM", "PHR", "GVR", "VCB", "HPG"],
    "FPT": ["CMG", "FRT", "MWG", "CTR", "VGI", "ELC", "ITD", "VCB", "TCB", "SSI"],
    "GAS": ["PVS", "PVD", "BSR", "POW", "PLX", "OIL", "PVC", "PVT", "VCB", "HPG"],
    "GVR": ["PHR", "DPR", "TRC", "BCM", "KBC", "IDC", "VGC", "VCB", "BID", "CTG"],
    "HDB": ["VIB", "TPB", "MSB", "OCB", "SSB", "LPB", "TCB", "MBB", "VPB", "VJC"],
    "HPG": ["HSG", "NKG", "SMC", "TLH", "POM", "VHM", "CTD", "HBC", "VCB", "BID"],
    "LPB": ["SHB", "SSB", "MSB", "OCB", "HDB", "VIB", "TPB", "STB", "MBB", "ACB"],
    "MBB": ["TCB", "VPB", "ACB", "STB", "HDB", "VIB", "TPB", "VCB", "CTG", "BID"],
    "MSN": ["VNM", "SAB", "MCH", "MML", "MSR", "MWG", "PNJ", "VCB", "TCB", "VIC"],
    "MWG": ["FRT", "DGW", "PET", "PNJ", "MSN", "VNM", "VCB", "TCB", "MBB", "SSI"],
    "PLX": ["OIL", "BSR", "GAS", "PVS", "PVD", "POW", "VCB", "BID", "CTG", "HPG"],
    "SAB": ["BHN", "VNM", "MSN", "KDC", "SCD", "SMB", "BSP", "VCB", "TCB", "SSI"],
    "SHB": ["SSB", "LPB", "MSB", "OCB", "HDB", "VIB", "TPB", "STB", "MBB", "ACB"],
    "SSB": ["SHB", "LPB", "MSB", "OCB", "HDB", "VIB", "TPB", "STB", "ABB", "VCB"],
    "SSI": ["VND", "VCI", "HCM", "SHS", "MBS", "VIX", "FTS", "BSI", "VCB", "TCB"],
    "STB": ["ACB", "MBB", "TCB", "VPB", "HDB", "VIB", "TPB", "VCB", "CTG", "BID"],
    "TCB": ["MBB", "VPB", "ACB", "STB", "HDB", "VIB", "TPB", "MSN", "VIC", "VHM"],
    "TPB": ["VIB", "HDB", "MSB", "OCB", "SSB", "LPB", "TCB", "MBB", "VPB", "FPT"],
    "VCB": ["BID", "CTG", "MBB", "TCB", "VPB", "ACB", "STB", "HPG", "VIC", "VHM"],
    "VHM": ["VIC", "VRE", "NVL", "KDH", "NLG", "PDR", "DIG", "DXG", "TCB", "MBB"],
    "VIB": ["HDB", "TPB", "MSB", "OCB", "SSB", "LPB", "TCB", "MBB", "VPB", "ACB"],
    "VIC": ["VHM", "VRE", "TCB", "MBB", "VPB", "HPG", "MSN", "VNM", "VCB", "BID"],
    "VJC": ["HVN", "AST", "SCS", "ACV", "HDB", "VCB", "BID", "CTG", "TCB", "MBB"],
    "VNM": ["MSN", "SAB", "KDC", "MCM", "MML", "MWG", "PNJ", "VCB", "TCB", "SSI"],
    "VPB": ["TCB", "MBB", "ACB", "STB", "HDB", "VIB", "TPB", "VCB", "CTG", "BID"],
    "VPL": ["VIC", "VHM", "VRE", "VJC", "HVN", "TCB", "MSN", "VNM", "VCB", "BID"],
    "VRE": ["VIC", "VHM", "NVL", "KDH", "NLG", "PDR", "TCB", "MBB", "VPB", "VCB"],
}


# ===================================================================
# 2. DATE HELPERS — lazy, tính lúc gọi không phải lúc import
# ===================================================================

def _date_range() -> tuple[datetime, datetime, str, str]:
    """Trả về (start_obj, end_obj, start_str, end_str) tính tại thời điểm gọi."""
    end   = datetime.today()
    start = end - relativedelta(years=3)
    return start, end, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def get_trading_dates() -> pd.Index:
    """
    Dùng trading dates thực tế từ features/ folder.
    Fallback về business days nếu chưa có features.
    Tính lại mỗi lần gọi → không bị stale.
    """
    _, _, start_str, end_str = _date_range()

    if os.path.exists(FEATURES_DIR):
        csv_files = [f for f in os.listdir(FEATURES_DIR) if f.endswith(".csv")]
        if csv_files:
            try:
                sample = pd.read_csv(os.path.join(FEATURES_DIR, csv_files[0]))
                if "time" in sample.columns:
                    return pd.Index(
                        pd.to_datetime(sample["time"]).dt.strftime("%Y-%m-%d")
                    )
            except Exception:
                pass

    log.warning("Không tìm thấy features/, dùng business days làm fallback.")
    return pd.Index(pd.bdate_range(start_str, end_str).strftime("%Y-%m-%d"))


# ===================================================================
# 3. CRAWL
# ===================================================================

def parse_date(raw_date: str) -> str | None:
    if not raw_date:
        return None
    raw_date = raw_date.strip().lower()
    if "trước" in raw_date or "hôm nay" in raw_date:
        return datetime.today().strftime("%Y-%m-%d")
    match = re.search(r"(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})", raw_date)
    if match:
        try:
            return datetime(
                int(match.group(3)), int(match.group(2)), int(match.group(1))
            ).strftime("%Y-%m-%d")
        except ValueError:
            pass
    return None


def parse_datetime(raw_date: str) -> tuple[str | None, str | None]:
    """
    Parse CafeF time label into (date_str, published_at_str).
    - date_str: YYYY-MM-DD
    - published_at_str: YYYY-MM-DD HH:MM:SS when HH:MM exists, otherwise YYYY-MM-DD
    """
    date_str = parse_date(raw_date)
    if not date_str:
        return None, None
    m = re.search(r"(\d{1,2})[:hH](\d{2})", str(raw_date))
    if m:
        hh = int(m.group(1))
        mm = int(m.group(2))
        if 0 <= hh <= 23 and 0 <= mm <= 59:
            dt = datetime.strptime(f"{date_str} {hh:02d}:{mm:02d}:00", "%Y-%m-%d %H:%M:%S")
            return date_str, dt.strftime("%Y-%m-%d %H:%M:%S")
    return date_str, date_str


def crawl_and_save_single_ticker(ticker: str, max_pages: int = 30) -> str:
    start_obj, end_obj, start_str, _ = _date_range()
    raw_path = os.path.join(RAW_DIR, f"{ticker}.csv")

    if os.path.exists(raw_path) and os.path.getsize(raw_path) > 10:
        log.info(f"[CRAWL] Skip {ticker} (đã có)")
        return ticker
    if os.path.exists(raw_path):
        os.remove(raw_path)

    articles = []
    for page in range(1, max_pages + 1):
        try:
            resp = requests.get(
                CAFEF_AJAX_URL,
                params={
                    "symbol": ticker, "floorID": "0", "configID": "0",
                    "PageIndex": str(page), "PageSize": "30", "Type": "2",
                },
                headers=HEADERS,
                timeout=10,
            )
            if not resp.text.strip() or "<li" not in resp.text:
                break

            soup = BeautifulSoup(resp.text, "html.parser")
            valid_in_page = 0

            for item in soup.find_all("li"):
                a_tag   = item.find("a", class_="docnhanhTitle") or item.find("a")
                span_tag = item.find("span", class_="timeTitle")
                if not a_tag or not span_tag:
                    continue
                title    = a_tag.get("title") or a_tag.get_text(strip=True)
                date_str, published_at = parse_datetime(span_tag.get_text(strip=True))
                if not date_str:
                    continue
                dt = datetime.strptime(date_str, "%Y-%m-%d")
                if start_obj <= dt <= end_obj:
                    articles.append(
                        {
                            "ticker": ticker,
                            "date": date_str,
                            "title": title,
                            "published_at": published_at,
                        }
                    )
                    valid_in_page += 1

            if valid_in_page == 0 and page > 1:
                log.debug(f"[CRAWL] {ticker} page {page}: không có bài trong range, dừng.")
                break

        except Exception as e:
            log.warning(f"[CRAWL] {ticker} page {page} lỗi: {e} — dừng crawl mã này.")
            break

    pd.DataFrame(articles).to_csv(raw_path, index=False)
    log.info(f"[CRAWL] {ticker}: {len(articles)} bài → {raw_path}")
    return ticker


def run_crawling_phase(tickers: list[str], max_workers: int = 8) -> None:
    log.info(f"[PHASE 1] Crawl {len(tickers)} mã, {max_workers} workers...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(crawl_and_save_single_ticker, t) for t in tickers]
        for _ in as_completed(futures):
            pass


# ===================================================================
# 4. CHẤM ĐIỂM SENTIMENT
# ===================================================================

def analyze_sentiment_batch(titles: list[str]) -> list[int]:
    """
    Trả về list số nguyên raw (0/1/2) cùng độ dài với titles.
    Encoding sang -1/0/1 thực hiện ở tầng caller.
    """
    prompt = (
        f"Đánh giá cảm xúc {len(titles)} tiêu đề tin tức chứng khoán tiếng Việt.\n"
        "Quy tắc: Tiêu cực=0, Tích cực=1, Trung lập=2.\n\n"
        f"Danh sách:\n{json.dumps(titles, ensure_ascii=False)}\n\n"
        f'Trả về JSON: {{"scores": [<{len(titles)} số nguyên>]}}\n'
        "KHÔNG giải thích, KHÔNG lặp lại tiêu đề."
    )

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Bạn chỉ xuất JSON. Không có văn bản khác."},
                    {"role": "user",   "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=300,
            )
            data   = json.loads(response.choices[0].message.content)
            scores = data.get("scores", [])
            if len(scores) == len(titles) and all(s in (0, 1, 2) for s in scores):
                return scores
            log.warning(
                f"[SENTIMENT] Lệch số lượng ({len(scores)}/{len(titles)}), thử lại {attempt+1}..."
            )
        except Exception as e:
            log.warning(f"[SENTIMENT] API lỗi lần {attempt+1}: {e}")
            time.sleep(2 ** attempt)

    log.error("[SENTIMENT] Batch thất bại sau 3 lần, gán trung lập.")
    return [2] * len(titles)


def score_and_save_single_ticker(ticker: str, batch_size: int = 20) -> None:
    trading_dates = get_trading_dates()   # ← lazy, tính lại mỗi lần gọi
    scored_path   = os.path.join(SCORED_DIR, f"{ticker}.csv")

    if os.path.exists(scored_path):
        log.info(f"[SCORE] Skip {ticker} (đã có)")
        return

    raw_path = os.path.join(RAW_DIR, f"{ticker}.csv")
    if not os.path.exists(raw_path):
        log.warning(f"[SCORE] {ticker}: không có raw file, bỏ qua.")
        return

    try:
        df = pd.read_csv(raw_path) if os.path.getsize(raw_path) > 10 else pd.DataFrame()
    except pd.errors.EmptyDataError:
        df = pd.DataFrame()

    if df.empty or "title" not in df.columns:
        df_daily = pd.Series(0.0, index=trading_dates, name=f"{ticker}_S")
        df_daily.index.name = "time"
        df_daily.to_csv(scored_path)
        log.info(f"[SCORE] {ticker}: không có tin, gán neutral (0.0)")
        return

    titles     = df["title"].tolist()
    raw_scores = []
    log.info(f"[SCORE] {ticker}: {len(titles)} bài, {(len(titles)-1)//batch_size+1} batch...")
    for i in range(0, len(titles), batch_size):
        batch = titles[i : i + batch_size]
        raw_scores.extend(analyze_sentiment_batch(batch))
        log.info(f"  {ticker}: {min(i+batch_size, len(titles))}/{len(titles)}")
        time.sleep(0.5)

    df["sentiment_score"] = [SENTIMENT_MAP[s] for s in raw_scores]
    df_daily = df.groupby("date")["sentiment_score"].mean()
    df_daily = df_daily.reindex(trading_dates).fillna(0.0)
    df_daily.name = f"{ticker}_S"
    df_daily.index.name = "time"
    df_daily.to_csv(scored_path)
    log.info(f"[SCORE] {ticker}: xong → {scored_path}")


def run_scoring_phase(tickers: list[str], max_workers: int = 4) -> None:
    log.info(f"[PHASE 2] Score {len(tickers)} mã, {max_workers} workers...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(score_and_save_single_ticker, t) for t in tickers]
        for _ in as_completed(futures):
            pass


# ===================================================================
# 5. MERGE FINAL
# ===================================================================

def run_merge_phase() -> None:
    trading_dates = get_trading_dates()   # ← lazy
    log.info("[PHASE 3] Merge dataset final cho 30 mã VN30...")

    for target in VN30_SYMBOLS:
        final_path = os.path.join(OUTPUT_DIR, f"{target}_Full_Sentiment.csv")
        if os.path.exists(final_path):
            log.info(f"[MERGE] Skip {target} (đã có)")
            continue

        target_score_file = os.path.join(SCORED_DIR, f"{target}.csv")
        if os.path.exists(target_score_file):
            df_final = pd.read_csv(target_score_file, index_col="time")
        else:
            df_final = pd.DataFrame(
                0.0, index=trading_dates, columns=[f"{target}_S"]
            )
            df_final.index.name = "time"

        for rel in VN30_RELATED_MAP[target]:
            rel_score_file = os.path.join(SCORED_DIR, f"{rel}.csv")
            if os.path.exists(rel_score_file):
                df_rel = pd.read_csv(rel_score_file, index_col="time")
                df_final = df_final.join(df_rel, how="left")
            else:
                df_final[f"{rel}_S"] = 0.0

        df_final = df_final.fillna(0.0)
        df_final.to_csv(final_path)
        log.info(f"[MERGE] {target}: {df_final.shape} → {final_path}")


# ===================================================================
# 6. MAIN
# ===================================================================

def main() -> None:
    _, _, start_str, end_str = _date_range()
    trading_dates = get_trading_dates()

    all_symbols: set[str] = set(VN30_SYMBOLS)
    for related_list in VN30_RELATED_MAP.values():
        all_symbols.update(related_list)
    all_symbols_list = sorted(all_symbols)

    log.info(f"==== BẮT ĐẦU PIPELINE: {len(all_symbols_list)} mã unique ====")
    log.info(f"Date range: {start_str} → {end_str}")
    log.info(f"Trading dates: {len(trading_dates)} ngày")

    run_crawling_phase(all_symbols_list, max_workers=8)
    run_scoring_phase(all_symbols_list, max_workers=4)
    run_merge_phase()

    log.info("==== PIPELINE HOÀN TẤT ====")


if __name__ == "__main__":
    main()
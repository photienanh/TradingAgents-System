from typing import Annotated
from datetime import datetime
from dateutil.relativedelta import relativedelta
from bs4 import BeautifulSoup
from email.utils import parsedate_to_datetime
import feedparser
import requests
import re


def _parse_input_date(date_str):
    if "-" in date_str:
        return datetime.strptime(date_str, "%Y-%m-%d")
    return datetime.strptime(date_str, "%m/%d/%Y")


def _is_cafef(url: str, source: str) -> bool:
    url_l = (url or "").lower()
    source_l = (source or "").lower()
    return "cafef.vn" in url_l or "cafef" in source_l


def _extract_article_text(html: str) -> str:
    soup = BeautifulSoup(html or "", "html.parser")
    for s in soup(["script", "style", "noscript"]):
        s.extract()

    selectors = [
        "div#mainContent",
        "div.knc-content",
        "div.detail-content",
        "div#CafeF_NoiDung",
        "article",
    ]

    node = None
    for selector in selectors:
        node = soup.select_one(selector)
        if node is not None:
            break

    node = node or soup
    paragraphs = [
        p.get_text(" ", strip=True)
        for p in node.select("p")
        if p.get_text(" ", strip=True)
    ]
    text = " ".join(paragraphs) if paragraphs else node.get_text(" ", strip=True)
    return re.sub(r"\s+", " ", text).strip()


def _summarize(text: str, max_chars: int = 420) -> str:
    text = re.sub(r"\s+", " ", (text or "")).strip()
    if not text:
        return ""

    parts = re.split(r"(?<=[.!?])\s+", text)
    selected = []
    total = 0
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if selected and total + len(part) + 1 > max_chars:
            break
        selected.append(part)
        total += len(part) + 1
        if len(selected) >= 3:
            break

    if not selected:
        return text[:max_chars].rstrip() + ("..." if len(text) > max_chars else "")
    return " ".join(selected)


def _fetch_cafef_summary(url: str) -> str:
    if "cafef.vn" not in (url or "").lower():
        return ""
    try:
        response = requests.get(
            url,
            timeout=15,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                )
            },
        )
    except Exception:
        return ""

    text = _extract_article_text(response.text)
    if len(text) < 180:
        return ""
    return _summarize(text)


def _query_tokens(query: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", (query or "")).strip().lower()
    if not normalized:
        return []
    return [t for t in normalized.split(" ") if len(t) >= 2]


def _extract_symbol(query: str) -> str:
    # Prefer ticker-like token (e.g. HPG, VNM, FPT) for CafeF symbol endpoint.
    q = (query or "").strip().upper()
    if not q:
        return ""
    first = q.split()[0]
    if re.fullmatch(r"[A-Z0-9]{2,10}", first):
        return first
    return ""


def _parse_cafef_deploy_date(value: str):
    # CafeF uses .NET date format: /Date(1773075600000)/
    if not value:
        return None
    m = re.search(r"/Date\((\d+)\)/", str(value))
    if not m:
        return None
    try:
        ts_ms = int(m.group(1))
        return datetime.fromtimestamp(ts_ms / 1000)
    except Exception:
        return None


def _matches_query(query: str, title: str, summary: str) -> bool:
    tokens = _query_tokens(query)
    if not tokens:
        return True
    haystack = f"{title} {summary}".lower()
    return any(t in haystack for t in tokens)


def _score_item(item: dict) -> int:
    # Prefer richer extracted summaries and longer informative titles.
    title_len = len(item.get("title", ""))
    snippet_len = len(item.get("snippet", ""))
    score = min(snippet_len, 500) + min(title_len, 120)
    query = (item.get("query", "") or "").lower()
    if query and query in (item.get("title", "") or "").lower():
        score += 80
    return score


def getNewsData(query, start_date, end_date, max_items=5):
    """Fetch CafeF-only news and summarize content from the original article pages."""
    start_dt = _parse_input_date(start_date)
    end_dt = _parse_input_date(end_date)

    symbol = _extract_symbol(query)
    candidates = []
    seen = set()

    if symbol:
        # Primary source: CafeF symbol news API.
        for page_index in range(1, 6):
            api_url = (
                "https://cafef.vn/du-lieu/Ajax/PageNew/News.ashx"
                f"?symbol={symbol.lower()}&NewsType=0&pageIndex={page_index}&pageSize=20"
            )
            try:
                response = requests.get(
                    api_url,
                    timeout=15,
                    headers={
                        "User-Agent": (
                            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                            "AppleWebKit/537.36 (KHTML, like Gecko) "
                            "Chrome/124.0.0.0 Safari/537.36"
                        )
                    },
                )
                payload = response.json()
            except Exception:
                continue

            data = payload.get("Data", []) if isinstance(payload, dict) else []
            if not data:
                break

            for item in data:
                title = str(item.get("Title", "")).strip()
                subtitle = str(item.get("SubTitle", "")).strip()
                link = str(item.get("LinkDetail", "")).strip()
                deploy_dt = _parse_cafef_deploy_date(item.get("DeployDate"))

                if not link:
                    continue
                if link.startswith("/"):
                    link = f"https://cafef.vn{link}"
                link = link.split("?")[0]

                if deploy_dt is not None and not (start_dt <= deploy_dt <= end_dt):
                    continue

                key = (link or title).lower()
                if key in seen:
                    continue
                seen.add(key)

                candidates.append(
                    {
                        "link": link,
                        "title": title,
                        "snippet": subtitle,
                        "date": deploy_dt.strftime("%Y-%m-%d %H:%M:%S") if deploy_dt else "",
                        "source": "CafeF",
                        "query": query,
                    }
                )

            # If we already have enough candidates, stop early.
            if len(candidates) >= max(20, max_items * 3):
                break

    if not candidates:
        # Fallback for non-ticker query: CafeF home RSS filtered by query/date.
        feed = feedparser.parse("https://cafef.vn/home.rss")

        for entry in getattr(feed, "entries", []):
            link = entry.get("link", "")
            title = entry.get("title", "")
            summary = BeautifulSoup(entry.get("summary", ""), "html.parser").get_text(
                " ", strip=True
            )
            date = entry.get("published", "")
            source = (
                entry.get("source", {}).get("title", "")
                if isinstance(entry.get("source"), dict)
                else ""
            )

            if not _is_cafef(link, source):
                continue
            if not _matches_query(query, title, summary):
                continue

            if date:
                try:
                    published_dt = parsedate_to_datetime(date)
                    published_naive = published_dt.replace(tzinfo=None)
                    if not (start_dt <= published_naive <= end_dt):
                        continue
                except Exception:
                    pass

            de_dup_key = (link or title).strip().lower()
            if not de_dup_key or de_dup_key in seen:
                continue
            seen.add(de_dup_key)

            title_l = title.lower()
            if any(k in title_l for k in ["trực tiếp", "livestream", "video"]):
                continue

            candidates.append(
                {
                    "link": link,
                    "title": title,
                    "snippet": summary,
                    "date": date,
                    "source": "CafeF",
                    "query": query,
                }
            )

    results = []
    for item in candidates[: max(20, max_items * 3)]:
        summary = _fetch_cafef_summary(item["link"])
        # Fallback to available subtitle/summary if page extraction fails.
        if not summary:
            summary = _summarize(item.get("snippet", ""), max_chars=280)
        if not summary:
            continue
        enriched = dict(item)
        enriched["snippet"] = summary
        results.append(enriched)

    results.sort(key=_score_item, reverse=True)
    return results[:max_items]

def get_cafef_news(
    query: Annotated[str, "Query to search with"],
    curr_date: Annotated[str, "Current date (yyyy-mm-dd) format"],
    look_back_days: Annotated[int, "Look-back days"] = 30,
) -> str:
    """Fetch CafeF news using current date and look-back days only."""
    query = query.strip()

    if look_back_days < 0:
        return "look_back_days phải >= 0"

    end_date = curr_date
    effective_lookback_days = max(look_back_days, 30)
    start_dt = datetime.strptime(curr_date, "%Y-%m-%d") - relativedelta(days=effective_lookback_days)
    start_date = start_dt.strftime("%Y-%m-%d")

    news_results = getNewsData(query, start_date, end_date)

    news_str = ""

    for news in news_results:
        news_str += (
            f"### {news['title']} (nguồn: {news['source']})\n"
            f"{news['link']}\n\n{news['snippet']}\n\n"
        )

    if len(news_results) == 0:
        return ""

    return f"## Tin tức về {query}, từ {start_date} đến {end_date}:\n\n{news_str}"

"""
tradingagents/dataflows/f247_forum.py

Scrape F247.com forum threads tagged with a stock ticker.
URL pattern: https://f247.com/tag/{ticker}?order=activity
"""

import re
import json
import html
import logging
from datetime import datetime, timedelta
from typing import Annotated

import requests
from bs4 import BeautifulSoup

import feedparser
from urllib.parse import quote_plus
from email.utils import parsedate_to_datetime

log = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.8",
}
_TIMEOUT = 15
_BASE = "https://f247.com"


# ── Date helpers ──────────────────────────────────────────────────────

_VN_MONTHS = {
    "Tháng Một": 1, "Tháng Hai": 2, "Tháng Ba": 3, "Tháng Tư": 4,
    "Tháng Năm": 5, "Tháng Sáu": 6, "Tháng Bảy": 7, "Tháng Tám": 8,
    "Tháng Chín": 9, "Tháng Mười": 10, "Tháng Mười Một": 11, "Tháng Mười Hai": 12,
}

def _parse_vn_date(text: str) -> datetime | None:
    """Parse 'Tháng Tư 20, 2026' or '05/04/2026' or '5 Thg 04 2026 09:56'."""
    text = (text or "").strip()
    if not text:
        return None

    # Format: "Tháng Tư 20, 2026"
    for vn_name, month_num in _VN_MONTHS.items():
        if vn_name in text:
            m = re.search(r"(\d{1,2}),?\s*(\d{4})", text)
            if m:
                try:
                    return datetime(int(m.group(2)), month_num, int(m.group(1)))
                except ValueError:
                    pass

    # Format: "5 Thg 04 2026 09:56"
    m = re.match(r"(\d{1,2})\s+Thg\s+(\d{2})\s+(\d{4})(?:\s+(\d{2}):(\d{2}))?", text)
    if m:
        try:
            day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
            hour = int(m.group(4)) if m.group(4) else 0
            minute = int(m.group(5)) if m.group(5) else 0
            return datetime(year, month, day, hour, minute)
        except ValueError:
            pass

    # Format: "05/04/2026"
    m = re.match(r"(\d{2})/(\d{2})/(\d{4})", text)
    if m:
        try:
            return datetime(int(m.group(3)), int(m.group(2)), int(m.group(1)))
        except ValueError:
            pass

    # Unix ms timestamp from data-time attribute
    m = re.match(r"^\d{13}$", text)
    if m:
        try:
            return datetime.fromtimestamp(int(text) / 1000)
        except Exception:
            pass

    return None


def _parse_post_date(post_el) -> datetime | None:
    """Extract post datetime from a .topic-post element."""
    el = post_el.select_one(".relative-date")
    if el:
        # Try title attribute first ("5 Thg 04 2026 09:56")
        dt = _parse_vn_date(el.get("title", ""))
        if dt:
            return dt
        # Try data-time (Unix ms)
        data_time = el.get("data-time", "")
        if data_time:
            try:
                return datetime.fromtimestamp(int(data_time) / 1000)
            except Exception:
                pass
        dt = _parse_vn_date(el.get_text(strip=True))
        if dt:
            return dt
    return None


def _parse_iso_datetime(text: str) -> datetime | None:
    """Parse ISO datetime string such as '2026-04-05T02:56:49.654Z'."""
    if not text:
        return None
    try:
        # Keep datetime naive for consistent comparisons with start/end_dt.
        return datetime.fromisoformat(text.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None


def _extract_posts_from_preloaded(soup: BeautifulSoup, thread_url: str) -> list[dict]:
    """Extract posts from Discourse data-preloaded JSON when topic-post DOM is not present."""
    preloaded_el = soup.select_one("#data-preloaded")
    if not preloaded_el:
        return []

    data_attr = preloaded_el.get("data-preloaded", "")
    if not data_attr:
        return []

    try:
        preloaded = json.loads(data_attr)
    except json.JSONDecodeError:
        try:
            preloaded = json.loads(html.unescape(data_attr))
        except json.JSONDecodeError:
            return []

    topic_id_match = re.search(r"/t/(?:[^/]+/)?(\d+)", thread_url)
    topic_key = f"topic_{topic_id_match.group(1)}" if topic_id_match else None

    topic_blob = preloaded.get(topic_key) if topic_key else None
    if not topic_blob:
        # Fallback: find first topic_* key.
        for key, value in preloaded.items():
            if key.startswith("topic_"):
                topic_blob = value
                break

    if not topic_blob:
        return []

    if isinstance(topic_blob, str):
        try:
            topic_data = json.loads(topic_blob)
        except json.JSONDecodeError:
            try:
                topic_data = json.loads(html.unescape(topic_blob))
            except json.JSONDecodeError:
                return []
    elif isinstance(topic_blob, dict):
        topic_data = topic_blob
    else:
        return []

    return topic_data.get("post_stream", {}).get("posts", [])

def _truncate_post(text: str, max_chars: int = 500) -> str:
    if len(text) <= max_chars:
        return text

    sentences = re.split(r'(?<=[.!?])\s+', text)
    if len(sentences) <= 1:
        return text[:max_chars] + "..."

    last = sentences[-1]
    head, total = [], 0
    for sent in sentences[:-1]:
        if total + len(sent) + 1 > max_chars:
            break
        head.append(sent)
        total += len(sent) + 1

    if not head:
        return text[:max_chars] + "..."

    return " ".join(head) + " ... " + last

# ── Thread list scraper ───────────────────────────────────────────────

def _fetch_thread_list(
    ticker: str,
    max_threads: int,
    start_dt: datetime,
    end_dt: datetime,
) -> list[dict]:
    """Fetch thread list from https://f247.com/tag/{ticker}?order=activity."""
    target_ticker = ticker.upper()
    url = f"{_BASE}/tag/{ticker.lower()}?order=activity"
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
        resp.raise_for_status()
    except Exception as e:
        log.warning(f"[F247] Failed to fetch tag page for {ticker}: {e}")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    rows = soup.select(".topic-list-item")

    threads = []
    for row in rows:
        # Keep only threads tagged with exactly one tag == requested ticker.
        row_tags = [
            tag.get_text(strip=True).upper()
            for tag in row.select(".discourse-tags a.discourse-tag")
            if tag.get_text(strip=True)
        ]
        if len(row_tags) != 1 or row_tags[0] != target_ticker:
            continue

        # Title & URL
        a = row.select_one("a.title.raw-link")
        if not a:
            continue
        title = a.get_text(strip=True)
        href = a.get("href", "")
        if not href.startswith("http"):
            href = _BASE + href

        # Last activity date (last <td> in the row)
        tds = row.select("td")
        activity_date = None
        if tds:
            activity_date = _parse_vn_date(tds[-1].get_text(strip=True))

        # Reply count
        replies_el = row.select_one("td.replies .posts")
        replies = 0
        if replies_el:
            m = re.search(r"\d+", replies_el.get_text(strip=True))
            replies = int(m.group(0)) if m else 0

        # Keep only threads with recent activity in requested date window.
        if activity_date is None or not (start_dt <= activity_date <= end_dt):
            continue

        threads.append({
            "title": title,
            "url": href,
            "activity_date": activity_date,
            "replies": replies,
        })

    threads.sort(key=lambda t: t["activity_date"], reverse=True)
    return threads[:max_threads]


# ── Thread content scraper ────────────────────────────────────────────

def _fetch_thread_posts(
    thread: dict,
    max_posts_per_thread: int,
) -> list[dict]:
    """Fetch posts from a single thread with selection based on reply count."""
    url = thread["url"]
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
        resp.raise_for_status()
    except Exception as e:
        log.warning(f"[F247] Failed to fetch thread {url}: {e}")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")

    posts = []
    parsed_posts: list[dict] = []
    raw_posts = _extract_posts_from_preloaded(soup, url)
    if raw_posts:
        for raw_post in raw_posts:
            username = raw_post.get("username") or "unknown"
            post_dt = _parse_iso_datetime(raw_post.get("created_at", ""))

            cooked_html = raw_post.get("cooked", "")
            if not cooked_html:
                continue

            content_soup = BeautifulSoup(cooked_html, "html.parser")
            content = re.sub(r"\s+", " ", content_soup.get_text(" ", strip=True)).strip()
            content = _truncate_post(content)
            if len(content) < 10:
                continue

            parsed_posts.append({
                "post_number": int(raw_post.get("post_number") or (len(parsed_posts) + 1)),
                "username": username,
                "date": post_dt.strftime("%Y-%m-%d %H:%M") if post_dt else "",
                "content": content,
                "dt": post_dt,
            })
    else:
        post_els = soup.select(".topic-post")
        for post_el in post_els:
            username_el = post_el.select_one(".names .username a")
            username = username_el.get_text(strip=True) if username_el else "unknown"

            post_dt = _parse_post_date(post_el)

            content_el = post_el.select_one(".cooked")
            if not content_el:
                continue
            content = re.sub(r"\s+", " ", content_el.get_text(" ", strip=True)).strip()
            content = _truncate_post(content)
            if len(content) < 10:
                continue

            parsed_posts.append({
                "post_number": len(parsed_posts) + 1,
                "username": username,
                "date": post_dt.strftime("%Y-%m-%d %H:%M") if post_dt else "",
                "content": content,
                "dt": post_dt,
            })

    if not parsed_posts:
        return []

    first_post = min(parsed_posts, key=lambda p: p.get("post_number", 10**9))
    thread["published_date"] = first_post.get("dt")

    target_count = max_posts_per_thread
    replies = int(thread.get("replies", 0) or 0)
    if replies <= 10:
        # Take the newest posts only (up to 10).
        selected_posts = sorted(
            parsed_posts,
            key=lambda p: (p.get("dt") is not None, p.get("dt") or datetime.min, p.get("post_number", 0)),
            reverse=True,
        )[:target_count]
        selected_posts.sort(key=lambda p: (p.get("dt") or datetime.min, p.get("post_number", 0)))
    else:
        remaining_posts = [p for p in parsed_posts if p is not first_post]
        # Pick 9 newest posts plus the first post.
        newest_posts = sorted(
            remaining_posts,
            key=lambda p: (p.get("dt") is not None, p.get("dt") or datetime.min, p.get("post_number", 0)),
            reverse=True,
        )[: max(target_count - 1, 0)]
        newest_posts.sort(key=lambda p: (p.get("dt") or datetime.min, p.get("post_number", 0)))
        selected_posts = [first_post] + newest_posts

    for p in selected_posts[:target_count]:
        posts.append({
            "username": p["username"],
            "date": p["date"],
            "content": p["content"],
        })

    return posts


# ── Format output ─────────────────────────────────────────────────────

def _format_thread(thread: dict, posts: list[dict]) -> str:
    """Format one thread as a conversation block."""
    published_date = thread.get("published_date")
    published_date_text = published_date.strftime("%Y-%m-%d") if isinstance(published_date, datetime) else "N/A"
    lines = [
        f"### Thread: {thread['title']}",
        f"URL: {thread['url']}",
        f"Ngày đăng: {published_date_text}",
        f"Số bài viết trong khoảng thời gian: {len(posts)}",
        "",
    ]
    for p in posts:
        lines.append(f"[{p['date']}] {p['username']}: {p['content']}")
    return "\n".join(lines)


# ── Public API ────────────────────────────────────────────────────────

def get_f247_forum_posts(
    ticker: Annotated[str, "Ticker symbol (e.g. MBB, HPG)"],
    curr_date: Annotated[str | None, "Current date yyyy-mm-dd"] = None,
    look_back_days: Annotated[int, "Number of days to look back"] = 30,
    max_threads: Annotated[int, "Maximum number of threads to fetch"] = 10,
    max_posts_per_thread: Annotated[int, "Maximum posts to extract per thread"] = 10,
) -> str:
    """
    Scrape F247.com forum discussions tagged with a stock ticker.
    Returns threads sorted by recent activity, with posts as conversation blocks.
    """

    try:
        base_dt = datetime.strptime(curr_date, "%Y-%m-%d") if curr_date else datetime.now()
    except ValueError as e:
        return f"Lỗi parse curr_date: {e}"

    end_dt = base_dt.replace(hour=23, minute=59, second=59, microsecond=0)
    start_dt = (end_dt - timedelta(days=look_back_days)).replace(hour=0, minute=0, second=0, microsecond=0)

    threads = _fetch_thread_list(ticker, max_threads, start_dt, end_dt)
    if not threads:
        return f"Không tìm thấy thread nào cho mã {ticker} trên F247."

    output_parts = [
        f"## Thảo luận F247 — {ticker.upper()} — {start_dt.strftime('%Y-%m-%d')} đến {end_dt.strftime('%Y-%m-%d')}\n"
        f"Tìm thấy {len(threads)} thread có hoạt động gần đây.\n"
    ]

    for thread in threads:
        posts = _fetch_thread_posts(thread, max_posts_per_thread)
        if posts:
            output_parts.append(_format_thread(thread, posts))

    if len(output_parts) == 1:
        return (
            f"Có {len(threads)} thread được tìm thấy nhưng không có bài viết nào "
            f"trong khoảng {start_dt.strftime('%Y-%m-%d')} đến {end_dt.strftime('%Y-%m-%d')}."
        )

    return "\n\n---\n\n".join(output_parts)

# ── Google News for Social Analyst ───────────────────────────────────────────
def _parse_rss_date(date_str: str) -> datetime | None:
    if not date_str:
        return None
    try:
        return parsedate_to_datetime(date_str).replace(tzinfo=None)
    except Exception:
        return None

def get_ticker_news(
    ticker: Annotated[str, "Ticker symbol (e.g. MBB, HPG)"],
    curr_date: Annotated[str | None, "Current date yyyy-mm-dd"] = None,
    look_back_days: Annotated[int, "Number of days to look back"] = 30,
    max_items: Annotated[int, "Maximum number of news items"] = 10,
) -> str:
    """
    Tìm tin tức về mã cổ phiếu trên Google News.
    Trả về tiêu đề + mô tả ngắn, giới hạn theo khoảng thời gian.
    """

    try:
        base_dt = datetime.strptime(curr_date, "%Y-%m-%d") if curr_date else datetime.now()
    except ValueError as e:
        return f"Lỗi parse curr_date: {e}"

    end_dt = base_dt.replace(hour=23, minute=59, second=59, microsecond=0)
    start_dt = (end_dt - timedelta(days=look_back_days)).replace(hour=0, minute=0, second=0, microsecond=0)

    query = quote_plus(f"{ticker} tin tức")
    cd_min = start_dt.strftime("%m/%d/%Y")
    cd_max = end_dt.strftime("%m/%d/%Y")
    url = (
        f"https://news.google.com/rss/search"
        f"?q={query}&hl=vi&gl=VN&ceid=VN:vi"
        f"&tbs=cdr:1,cd_min:{cd_min},cd_max:{cd_max}"
    )

    try:
        feed = feedparser.parse(url)
    except Exception as e:
        return f"Lỗi khi tải Google News: {e}"

    items = []
    for entry in feed.entries:
        pub_dt = _parse_rss_date(entry.get("published", ""))
        if pub_dt and not (start_dt <= pub_dt <= end_dt):
            continue

        title = entry.get("title", "").strip()
        source = entry.get("source", {}).get("title", "") if isinstance(entry.get("source"), dict) else ""
        description = BeautifulSoup(entry.get("summary", ""), "html.parser").get_text(" ", strip=True)
        pub_str = pub_dt.strftime("%Y-%m-%d") if pub_dt else ""

        items.append(f"### {title}\nNguồn: {source} | Ngày: {pub_str}\n{description}")

        if len(items) >= max_items:
            break

    if not items:
        return (
            f"Không tìm thấy tin tức nào cho {ticker} trong khoảng "
            f"{start_dt.strftime('%Y-%m-%d')} đến {end_dt.strftime('%Y-%m-%d')}."
        )

    header = f"## Tin tức — {ticker.upper()} — {start_dt.strftime('%Y-%m-%d')} đến {end_dt.strftime('%Y-%m-%d')}\n"
    return header + "\n\n---\n\n".join(items)
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
from html import unescape
from typing import Annotated
import re
import xml.etree.ElementTree as ET

import requests


VIETSTOCK_GLOBAL_RSS_FEEDS = [
	"https://vietstock.vn/768/kinh-te/kinh-te-dau-tu.rss",
	"https://vietstock.vn/772/the-gioi/tai-chinh-quoc-te.rss",
	"https://vietstock.vn/775/the-gioi/kinh-te-nganh.rss",
]
VIETSTOCK_ITEMS_PER_FEED = 5
VIETSTOCK_TOTAL_ITEMS_MIN = VIETSTOCK_ITEMS_PER_FEED * len(VIETSTOCK_GLOBAL_RSS_FEEDS)


def _strip_html(text: str) -> str:
	raw = unescape(text or "")
	raw = re.sub(r"<img[^>]*>", " ", raw, flags=re.IGNORECASE)
	raw = re.sub(r"<[^>]+>", " ", raw)
	return re.sub(r"\s+", " ", raw).strip()


def _parse_pub_date(value: str):
	if not value:
		return None
	try:
		return parsedate_to_datetime(value)
	except Exception:
		return None


def _fetch_feed_items(feed_url: str):
	response = requests.get(
		feed_url,
		timeout=20,
		headers={
			"User-Agent": (
				"Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
				"AppleWebKit/537.36 (KHTML, like Gecko) "
				"Chrome/124.0.0.0 Safari/537.36"
			),
			"Accept": "application/rss+xml, application/xml, text/xml, */*",
			"Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
			"Referer": "https://vietstock.vn/",
		},
	)
	response.raise_for_status()
	response.encoding = response.encoding or "utf-8"

	root = ET.fromstring(response.text)
	channel = root.find("channel")
	if channel is None:
		return []

	items = []
	for item in channel.findall("item"):
		title = (item.findtext("title") or "").strip()
		description_html = item.findtext("description") or ""
		pub_date_raw = (item.findtext("pubDate") or "").strip()
		pub_date = _parse_pub_date(pub_date_raw)

		if not title or not description_html:
			continue

		summary = _strip_html(description_html)
		if not summary:
			continue

		items.append(
			{
				"title": title,
				"summary": summary,
				"pub_date": pub_date,
			}
		)

	return items


def get_vietstock_global_news(
	curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
	look_back_days: Annotated[int, "Number of days to look back"] = 7,
	limit: Annotated[int, "Maximum number of articles to return"] = 15,
) -> str:
	curr_dt = datetime.strptime(curr_date, "%Y-%m-%d")
	start_dt = curr_dt - timedelta(days=look_back_days)
	end_dt = curr_dt.replace(hour=23, minute=59, second=59, microsecond=999999)

	all_items = []
	for feed_url in VIETSTOCK_GLOBAL_RSS_FEEDS:
		try:
			feed_items = _fetch_feed_items(feed_url)
			all_items.extend(feed_items[:VIETSTOCK_ITEMS_PER_FEED])
		except Exception:
			continue

	filtered_items = []
	for item in all_items:
		pub_date = item.get("pub_date")
		if pub_date is not None:
			pub_naive = pub_date.replace(tzinfo=None)
			if not (start_dt <= pub_naive <= end_dt):
				continue
		filtered_items.append(item)

	filtered_items.sort(
		key=lambda x: x.get("pub_date") or datetime.min,
		reverse=True,
	)

	effective_limit = VIETSTOCK_TOTAL_ITEMS_MIN
	if limit > 0:
		effective_limit = max(limit, VIETSTOCK_TOTAL_ITEMS_MIN)
	filtered_items = filtered_items[:effective_limit]

	if not filtered_items:
		return ""

	start_date_str = start_dt.date().isoformat()
	end_date_str = curr_dt.date().isoformat()
	lines = [
		f"## Tin tức vĩ mô tổng hợp (từ {start_date_str} đến {end_date_str}):",
		"",
	]
	for idx, item in enumerate(filtered_items, 1):
		lines.append(f"### {idx}. {item['title']}")
		lines.append(f"- Tóm tắt: {item['summary']}")
		lines.append("")

	return "\n".join(lines).strip()

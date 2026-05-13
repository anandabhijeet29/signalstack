import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

import feedparser
import requests

from signalstack.models.article import Article

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X) SignalStack/1.0"
}


def _fetch_single_feed(feed_url: str, max_entries: int) -> List[Article]:
    articles: List[Article] = []
    logger.debug("Fetching feed: %s", feed_url)

    response = None
    for attempt in range(3):
        try:
            response = requests.get(feed_url, headers=HEADERS, timeout=10)
            break
        except requests.exceptions.RequestException:
            if attempt < 2:
                logger.debug("Retrying %s...", feed_url)
            else:
                raise

    if response is None:
        raise requests.exceptions.RequestException(
            "No response received after retries."
        )

    logger.debug("HTTP %d from %s", response.status_code, feed_url)
    response.raise_for_status()
    parsed_feed = feedparser.parse(response.text)

    source = parsed_feed.feed.get("title", "")
    entries = parsed_feed.entries[:max_entries]
    if not entries:
        logger.debug("No entries found for %s", feed_url)
        return articles

    logger.debug("%d entries found in %s", len(entries), feed_url)

    for entry in entries:
        articles.append(
            Article(
                title=getattr(entry, "title", "") or entry.get("title", ""),
                link=getattr(entry, "link", "") or entry.get("link", ""),
                source=source or "Unknown",
                summary=getattr(entry, "summary", None) or entry.get("summary"),
            )
        )

    return articles


def fetch_articles(feed_urls: List[str], max_entries: int = 5) -> List[Article]:
    articles: List[Article] = []

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(_fetch_single_feed, url, max_entries): url
            for url in feed_urls
        }
        for future in as_completed(futures):
            url = futures[future]
            try:
                articles.extend(future.result())
            except Exception:
                logger.warning("Failed to fetch feed: %s", url)

    return articles

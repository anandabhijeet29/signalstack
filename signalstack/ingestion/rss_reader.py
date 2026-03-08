from typing import Dict, List

import feedparser
import requests


HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X) SignalStack/1.0"
}


def fetch_articles(feed_urls: List[str]) -> List[Dict]:
    articles: List[Dict] = []

    for feed_url in feed_urls:
        print(f"Fetching feed: {feed_url}")

        try:
            response = None
            for attempt in range(3):
                try:
                    response = requests.get(feed_url, headers=HEADERS, timeout=10)
                    break
                except requests.exceptions.RequestException:
                    if attempt < 2:
                        print("Retrying...")
                    else:
                        raise

            if response is None:
                raise requests.exceptions.RequestException(
                    "No response received after retries."
                )

            print(f"HTTP status code: {response.status_code}")
            response.raise_for_status()
            parsed_feed = feedparser.parse(response.text)
        except Exception:
            print(f"Failed to fetch feed: {feed_url}")
            continue

        source = parsed_feed.feed.get("title", "")
        entries = parsed_feed.entries[:5]
        if not entries:
            print(f"No entries found for {feed_url}")
            continue

        print(f"number of entries found: {len(entries)}")

        for entry in entries:
            articles.append(
                {
                    "title": entry.get("title", ""),
                    "link": entry.get("link", ""),
                    "source": source,
                    "summary": entry.get("summary", ""),
                }
            )

    return articles

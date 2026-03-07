from typing import Dict, List
import feedparser


def fetch_articles(feed_urls: List[str]) -> List[Dict]:
    articles: List[Dict] = []

    for feed_url in feed_urls:
        parsed_feed = feedparser.parse(feed_url)
        source = parsed_feed.feed.get("title", "")

        for entry in parsed_feed.entries[:5]:
            articles.append(
                {
                    "title": entry.get("title", ""),
                    "link": entry.get("link", ""),
                    "source": source,
                }
            )

    return articles

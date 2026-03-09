from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from signalstack.agents.ranker import rank_articles
from signalstack.ingestion.feed_loader import load_feeds
from signalstack.ingestion.rss_reader import fetch_articles
from signalstack.models.article import Article
from signalstack.processing.article_store import (
    filter_new_articles,
    load_seen_urls,
    save_seen_urls,
    update_seen_urls,
)
from signalstack.processing.extractor import extract_article_text


@dataclass
class PipelineConfig:
    top_n: int = 5
    max_age_days: int = 7
    min_content_length: int = 300


def run_pipeline(config: Optional[PipelineConfig] = None) -> List[Article]:
    print("Starting SignalStack pipeline")
    cfg = config or PipelineConfig()

    data_dir = Path(__file__).resolve().parent / "data"
    feeds_path = data_dir / "feeds.yaml"
    seen_store_path = data_dir / "seen_articles.json"

    feed_urls = load_feeds(str(feeds_path))
    articles = fetch_articles(feed_urls)
    print(f"Fetched {len(articles)} articles")

    if not articles:
        print("No articles fetched. Check network or feed availability.")
        return []

    seen_urls = load_seen_urls(str(seen_store_path), max_age_days=cfg.max_age_days)
    new_articles = filter_new_articles(articles, seen_urls)
    filtered_count = len(articles) - len(new_articles)
    print(f"Filtered {filtered_count} seen articles")
    print(f"{len(new_articles)} new articles remaining")

    if len(new_articles) == 0:
        print("No new articles today")
        return []

    print("Ranking articles")
    top_articles = rank_articles(new_articles, top_n=cfg.top_n)

    print("Extracting article content")
    for article in top_articles:
        try:
            extracted = extract_article_text(article.link)
            if extracted and len(extracted) < cfg.min_content_length:
                print(f"Content below min length for article: {article.link}")
                article.content = None
            else:
                article.content = extracted
        except Exception:
            print(f"Extraction failed for article: {article.link}")
            continue

    updated_seen_urls = update_seen_urls(new_articles, seen_urls)
    save_seen_urls(str(seen_store_path), updated_seen_urls)

    print("Pipeline complete")
    return top_articles

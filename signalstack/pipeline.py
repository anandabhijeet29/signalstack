from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from signalstack.agents.ranker import rank_articles
from signalstack.agents.summarizer import summarize_article
from signalstack.digest.generator import generate_digest, save_digest
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
    candidate_pool_size: int = 10
    max_age_days: int = 7
    min_content_length: int = 300
    vault_path: Optional[str] = None


def run_pipeline(
    config: Optional[PipelineConfig] = None,
) -> Tuple[List[Article], List[Dict]]:
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
        return [], []

    seen_urls = load_seen_urls(str(seen_store_path), max_age_days=cfg.max_age_days)
    new_articles = filter_new_articles(articles, seen_urls)
    filtered_count = len(articles) - len(new_articles)
    print(f"Filtered {filtered_count} seen articles")
    print(f"{len(new_articles)} new articles remaining")

    if len(new_articles) == 0:
        print("No new articles today")
        return [], []

    print("Extracting previews for ranking...")
    for article in new_articles:
        try:
            preview_text = extract_article_text(article.link)
            if preview_text:
                article.preview = preview_text[:1500]
            else:
                article.preview = article.summary
        except Exception:
            article.preview = article.summary

    print("Ranking articles")
    candidate_count = max(cfg.candidate_pool_size, cfg.top_n)
    ranked_articles = rank_articles(new_articles, top_n=candidate_count)
    print(f"Ranking complete: {len(ranked_articles)} articles selected")

    print("Generating summaries")
    summaries: List[Dict] = []
    summarized_articles: List[Article] = []
    print(f"Attempting summarization from {len(ranked_articles)} candidate articles")
    for article in ranked_articles:
        try:
            extracted = extract_article_text(article.link)
            if extracted and len(extracted) >= cfg.min_content_length:
                article.content = extracted
            else:
                article.content = None
            summary = summarize_article(article)
            if summary:
                summaries.append(summary)
                summarized_articles.append(article)
                if len(summaries) == cfg.top_n:
                    break
        except Exception:
            print(f"Summarization failed for article: {article.link}")
            continue
    print(f"{len(summaries)} summaries successfully generated")

    updated_seen_urls = update_seen_urls(new_articles, seen_urls)
    save_seen_urls(str(seen_store_path), updated_seen_urls)

    if summaries:
        print("Generating intelligence digest...")
        markdown = generate_digest(summaries)
        vault_path = (
            cfg.vault_path
            or "/Users/abhijeetanand/Documents/Obsidian Vault/Intelligence/SignalStack"
        )
        save_digest(markdown, vault_path)
    else:
        print("No summaries generated. Digest skipped.")
        return [], []

    print("Pipeline complete")
    return summarized_articles, summaries

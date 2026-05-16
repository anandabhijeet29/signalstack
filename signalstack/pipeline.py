import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from signalstack.agents.ranker import rank_articles
from signalstack.agents.summarizer import summarize_article
from signalstack.agents.synthesizer import synthesize_themes
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
from signalstack.agents.investigator import InvestigatorAgent
from signalstack.agents.debate_agent import run_text_debate
from signalstack.processing.extractor import extract_articles_concurrent

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    top_n: int = 5
    candidate_pool_size: int = 10
    max_age_days: int = 7
    min_content_length: int = 300
    max_entries_per_feed: int = 5
    vault_path: Optional[str] = None
    investigate: bool = False
    debate: bool = False
    debate_rounds: int = 3
    max_steps: int = 5
    max_urls: int = 10


def run_pipeline(
    config: Optional[PipelineConfig] = None,
) -> Tuple[List[Article], List[Dict]]:
    logger.info("Starting SignalStack pipeline")
    cfg = config or PipelineConfig()

    data_dir = Path(__file__).resolve().parent / "data"
    feeds_path = data_dir / "feeds.yaml"
    seen_store_path = data_dir / "seen_articles.json"

    feed_urls = load_feeds(str(feeds_path))
    articles = fetch_articles(feed_urls, max_entries=cfg.max_entries_per_feed)
    logger.info("Fetched %d articles", len(articles))

    if not articles:
        logger.warning("No articles fetched. Check network or feed availability.")
        return [], []

    seen_urls = load_seen_urls(str(seen_store_path), max_age_days=cfg.max_age_days)
    new_articles = filter_new_articles(articles, seen_urls)
    filtered_count = len(articles) - len(new_articles)
    logger.info("Filtered %d seen articles, %d new remaining", filtered_count, len(new_articles))

    if len(new_articles) == 0:
        logger.info("No new articles today")
        return [], []

    candidate_count = max(cfg.candidate_pool_size, cfg.top_n)
    for article in new_articles:
        article.preview = article.summary

    logger.info("Initial ranking of articles")
    initial_ranked = rank_articles(new_articles, top_n=candidate_count)
    logger.info("Initial ranking complete: %d candidates selected", len(initial_ranked))

    logger.info("Extracting previews for top candidates")
    urls = [a.link for a in initial_ranked]
    extraction_cache = extract_articles_concurrent(urls, min_content_length=cfg.min_content_length)
    for article in initial_ranked:
        extracted = extraction_cache.get(article.link)
        if extracted:
            article.preview = extracted[:1500]
        else:
            article.preview = article.summary

    logger.info("Re-ranking with extracted previews")
    ranked_articles = rank_articles(initial_ranked, top_n=candidate_count)
    logger.info("Ranking complete: %d articles selected", len(ranked_articles))

    logger.info("Generating summaries")
    summaries: List[Dict] = []
    summarized_articles: List[Article] = []
    logger.debug("Attempting summarization from %d candidate articles", len(ranked_articles))
    for article in ranked_articles:
        try:
            extracted = extraction_cache.get(article.link)
            if extracted and len(extracted) >= cfg.min_content_length:
                article.content = extracted
            else:
                article.content = None
            summary = summarize_article(article, min_content_length=cfg.min_content_length)
            if summary:
                summaries.append(summary)
                summarized_articles.append(article)
                if len(summaries) == cfg.top_n:
                    break
        except Exception:
            logger.warning("Summarization failed for article: %s", article.link)
            continue
    logger.info("%d summaries successfully generated", len(summaries))

    updated_seen_urls = update_seen_urls(summarized_articles, seen_urls)
    save_seen_urls(str(seen_store_path), updated_seen_urls)

    if summaries:
        themes = synthesize_themes(summaries)
        if themes:
            logger.info("Synthesized %d cross-article themes", len(themes))
        else:
            logger.warning("Theme synthesis failed or returned no themes")

        investigation_log: Optional[str] = None
        trace = None
        if cfg.investigate:
            logger.info("Running agentic investigation")
            agent = InvestigatorAgent(
                summaries,
                max_steps=cfg.max_steps,
                max_urls=cfg.max_urls,
            )
            trace = agent.investigate()
            if trace:
                log_md = trace.to_markdown()
                investigation_log = log_md if log_md else None
                if investigation_log:
                    logger.info("Investigation log generated (%d chars)", len(investigation_log))
                else:
                    logger.warning("Investigation produced no log content")
            else:
                logger.warning("Investigation returned no trace")

        debate_transcript: Optional[str] = None
        if cfg.debate:
            logger.info("Running text debate")
            debate_transcript = run_text_debate(
                summaries,
                trace=trace,
                rounds=cfg.debate_rounds,
            )
            if debate_transcript:
                logger.info("Debate transcript generated (%d chars)", len(debate_transcript))
            else:
                logger.warning("Debate returned no transcript")

        logger.info("Generating intelligence digest")
        markdown = generate_digest(
            summaries,
            themes=themes or [],
            investigation_log=investigation_log,
            debate_transcript=debate_transcript,
        )
        if cfg.vault_path:
            save_digest(markdown, cfg.vault_path)
        else:
            print(markdown)
    else:
        logger.warning("No summaries generated. Digest skipped.")
        return [], []

    logger.info("Pipeline complete")
    return summarized_articles, summaries

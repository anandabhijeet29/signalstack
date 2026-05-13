import datetime
import json
import logging
from pathlib import Path
from typing import Dict, List, Set

from signalstack.models.article import Article

logger = logging.getLogger(__name__)


def load_seen_urls(path: str, max_age_days: int = 7) -> Dict[str, str]:
    store_path = Path(path)
    now_utc = datetime.datetime.now(datetime.timezone.utc)

    if not store_path.exists():
        logger.debug("Seen store not found, starting fresh")
        return {}

    try:
        raw_data = json.loads(store_path.read_text(encoding="utf-8"))
        seen_urls = raw_data.get("seen_urls", {})
        if not isinstance(seen_urls, dict):
            raise ValueError("Invalid seen_urls structure")
    except Exception:
        logger.warning("Seen store corrupted, falling back to empty")
        return {}

    cleaned_seen_urls: Dict[str, str] = {}
    removed_expired = 0
    cutoff = now_utc - datetime.timedelta(days=max_age_days)

    for url, seen_at in seen_urls.items():
        if not isinstance(url, str) or not isinstance(seen_at, str):
            removed_expired += 1
            continue

        try:
            seen_dt = datetime.datetime.fromisoformat(seen_at)
        except ValueError:
            removed_expired += 1
            continue

        if seen_dt.tzinfo is None:
            seen_dt = seen_dt.replace(tzinfo=datetime.timezone.utc)

        if seen_dt < cutoff:
            removed_expired += 1
            continue

        cleaned_seen_urls[url] = seen_at

    logger.debug("Loaded %d seen URLs, removed %d expired", len(cleaned_seen_urls), removed_expired)
    return cleaned_seen_urls


def save_seen_urls(path: str, seen_urls: Dict[str, str]) -> None:
    store_path = Path(path)
    store_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {"seen_urls": seen_urls}
    store_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.debug("Saved %d seen URLs", len(seen_urls))


def filter_new_articles(
    articles: List[Article], seen_urls: Dict[str, str]
) -> List[Article]:
    seen_url_set: Set[str] = set(seen_urls.keys())
    new_articles: List[Article] = []

    for article in articles:
        link = article.link.strip()
        if not link or link in seen_url_set:
            continue

        new_articles.append(article)
        seen_url_set.add(link)

    return new_articles


def update_seen_urls(
    articles: List[Article], seen_urls: Dict[str, str]
) -> Dict[str, str]:
    updated_seen_urls = dict(seen_urls)
    timestamp = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat()

    for article in articles:
        link = article.link.strip()
        if link and link not in updated_seen_urls:
            updated_seen_urls[link] = timestamp

    return updated_seen_urls

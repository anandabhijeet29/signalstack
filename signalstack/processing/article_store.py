import json
import datetime
from typing import Dict, List, Set
from pathlib import Path

from signalstack.models.article import Article


def load_seen_urls(path: str, max_age_days: int = 7) -> Dict[str, str]:
    store_path = Path(path)
    now_utc = datetime.datetime.now(datetime.timezone.utc)

    if not store_path.exists():
        print("Seen store not found. Starting with empty seen URLs.")
        print("Loaded 0 seen URLs")
        print("Removed 0 expired URLs from seen store")
        return {}

    try:
        raw_data = json.loads(store_path.read_text(encoding="utf-8"))
        seen_urls = raw_data.get("seen_urls", {})
        if not isinstance(seen_urls, dict):
            raise ValueError("Invalid seen_urls structure")
    except Exception:
        print("Seen store missing/corrupted. Falling back to empty seen URLs.")
        print("Loaded 0 seen URLs")
        print("Removed 0 expired URLs from seen store")
        return {}

    loaded_count = len(seen_urls)
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

    print(f"Loaded {loaded_count} seen URLs")
    print(f"Removed {removed_expired} expired URLs from seen store")
    return cleaned_seen_urls


def save_seen_urls(path: str, seen_urls: Dict[str, str]) -> None:
    store_path = Path(path)
    store_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {"seen_urls": seen_urls}
    store_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved {len(seen_urls)} seen URLs")


def filter_new_articles(
    articles: List[Article], seen_urls: Dict[str, str]
) -> List[Article]:
    seen_url_set: Set[str] = set(seen_urls.keys())
    filtered_count = 0
    new_articles: List[Article] = []

    for article in articles:
        link = article.link.strip()
        if not link:
            filtered_count += 1
            continue

        if link in seen_url_set:
            filtered_count += 1
            continue

        new_articles.append(article)
        seen_url_set.add(link)

    print(f"Filtered {filtered_count} articles")
    print(f"New articles remaining: {len(new_articles)}")
    return new_articles


def update_seen_urls(
    articles: List[Article], seen_urls: Dict[str, str]
) -> Dict[str, str]:
    updated_seen_urls = dict(seen_urls)
    timestamp = datetime.datetime.now().replace(microsecond=0).isoformat()

    for article in articles:
        link = article.link.strip()
        if link and link not in updated_seen_urls:
            updated_seen_urls[link] = timestamp

    return updated_seen_urls

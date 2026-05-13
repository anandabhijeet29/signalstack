import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

import trafilatura

logger = logging.getLogger(__name__)


def looks_like_text(text: str) -> bool:
    if not text:
        return False

    printable_chars = sum(c.isprintable() for c in text)
    ratio = printable_chars / len(text)
    return ratio > 0.85


def extract_article_text(url: str, min_content_length: int = 300) -> Optional[str]:
    logger.debug("Extracting article: %s", url)

    try:
        downloaded = trafilatura.fetch_url(url)
    except Exception:
        logger.debug("Failed to download page: %s", url)
        return None

    if not downloaded:
        logger.debug("Failed to download page: %s", url)
        return None

    try:
        text = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=False,
        )
    except Exception:
        logger.debug("Extraction returned no content: %s", url)
        return None

    if text is None:
        logger.debug("Extraction returned no content: %s", url)
        return None

    if not looks_like_text(text):
        logger.debug("Corrupted or binary content detected: %s", url)
        return None

    cleaned_text = text.strip()
    if len(cleaned_text) < min_content_length:
        logger.debug("Content too short (%d chars): %s", len(cleaned_text), url)
        return None

    logger.debug("Extracted %d characters from %s", len(cleaned_text), url)
    return cleaned_text


def extract_articles_concurrent(
    urls: List[str], min_content_length: int = 300
) -> Dict[str, Optional[str]]:
    results: Dict[str, Optional[str]] = {}

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(extract_article_text, url, min_content_length): url
            for url in urls
        }
        for future in as_completed(futures):
            url = futures[future]
            try:
                results[url] = future.result()
            except Exception:
                results[url] = None

    return results

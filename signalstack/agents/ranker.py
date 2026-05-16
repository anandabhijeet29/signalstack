from __future__ import annotations

import logging
import re
from typing import List

from signalstack.models.article import Article

logger = logging.getLogger(__name__)

KEYWORDS = (
    "ai",
    "model",
    "agent",
    "compute",
    "startup",
    "policy",
    "research",
    "product",
    "strategy"
)


def _score_article(article: Article) -> float:
    title = str(article.title or "").strip()
    summary = str(article.summary or "").strip()
    preview = str(article.preview or "").strip()
    searchable_text = f"{title} {summary} {preview}".lower()

    title_length_score = min(len(title), 120) / 12.0

    keyword_hits = 0
    for keyword in KEYWORDS:
        if re.search(rf"\b{re.escape(keyword)}\b", searchable_text):
            keyword_hits += 1
    keyword_score = keyword_hits * 3.0

    return title_length_score + keyword_score


def rank_articles(articles: List[Article], top_n: int = 5) -> List[Article]:
    if top_n <= 0 or not articles:
        return []

    ranked = sorted(
        articles,
        key=lambda a: (-_score_article(a), str(a.title or "").lower()),
    )
    result = ranked[:top_n]

    logger.debug("Ranked %d articles, returning top %d", len(articles), len(result))
    return result

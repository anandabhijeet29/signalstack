from __future__ import annotations

from typing import List
import re

from signalstack.models.article import Article


KEYWORDS = (
    "ai",
    "model",
    "agent",
    "compute",
    "startup",
    "policy",
    "research",
)


def _score_article(article: Article) -> float:
    title = article.title.strip()
    summary = (article.summary or "").strip()
    searchable_text = f"{title} {summary}".lower()

    # Reward clear, descriptive titles without letting very long titles dominate.
    title_length_score = min(len(title), 120) / 12.0

    keyword_hits = 0
    for keyword in KEYWORDS:
        if re.search(rf"\b{re.escape(keyword)}\b", searchable_text):
            keyword_hits += 1
    keyword_score = keyword_hits * 3.0

    return title_length_score + keyword_score


def rank_articles(articles: List[Article], top_n: int = 5) -> List[Article]:
    print(f"Ranker received {len(articles)} articles")

    if top_n <= 0 or not articles:
        print("Ranker returning 0 articles")
        return []

    ranked = sorted(
        articles,
        key=lambda article: (-_score_article(article), article.title.lower()),
    )
    result = ranked[:top_n]

    print(f"Ranker returning {len(result)} articles")
    return result

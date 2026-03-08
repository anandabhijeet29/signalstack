from __future__ import annotations

from typing import Dict, List
import re


KEYWORDS = (
    "ai",
    "model",
    "agent",
    "compute",
    "startup",
    "policy",
    "research",
)


def _score_article(article: Dict) -> float:
    title = str(article.get("title", "")).strip()
    summary = str(article.get("summary", "")).strip()
    searchable_text = f"{title} {summary}".lower()

    # Reward clear, descriptive titles without letting very long titles dominate.
    title_length_score = min(len(title), 120) / 12.0

    keyword_hits = 0
    for keyword in KEYWORDS:
        if re.search(rf"\b{re.escape(keyword)}\b", searchable_text):
            keyword_hits += 1
    keyword_score = keyword_hits * 3.0

    return title_length_score + keyword_score


def rank_articles(articles: List[Dict], top_n: int = 5) -> List[Dict]:
    print(f"Ranker received {len(articles)} articles")

    if top_n <= 0 or not articles:
        print("Ranker returning 0 articles")
        return []

    scored_articles = []
    for article in articles:
        scored = dict(article)
        scored["score"] = _score_article(article)
        scored_articles.append(scored)

    ranked = sorted(
        scored_articles,
        key=lambda a: (-a["score"], str(a.get("title", "")).lower()),
    )
    result = ranked[:top_n]

    print(f"Ranker returning {len(result)} articles")
    return result

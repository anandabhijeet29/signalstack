import json
import logging
import os
from typing import Dict, Optional

from anthropic import Anthropic

from signalstack.models.article import Article

logger = logging.getLogger(__name__)

MODEL_NAME = os.getenv("ANTHROPIC_MODEL", "claude-haiku-3-5")

client: Optional[Anthropic] = None


def _get_client() -> Optional[Anthropic]:
    global client
    if client is not None:
        return client

    try:
        client = Anthropic()
    except Exception:
        client = None
    return client

SYSTEM_PROMPT = (
    "You are an analyst creating intelligence briefings from technology and AI "
    "articles. Your task is to extract the most important signal from the article "
    "and explain its implications."
)

USER_PROMPT_TEMPLATE = """
Analyze the following article and return structured JSON.

Focus on:
- the core thesis of the article
- implications for AI, technology, startups, or policy
- the most important insights

Return valid JSON with the following fields:

tldr: One concise sentence capturing the main idea.

summary: A 2-3 sentence explanation of the article's argument.

key_insights: A list of 3-5 important insights derived from the article.

importance_score: An integer from 1 to 10 representing how important the article is for an AI/technology intelligence digest.

Scoring guidelines:

1-3 -> minor commentary or opinion
4-6 -> moderately interesting analysis
7-8 -> important development or insight
9-10 -> major signal with strategic implications
"""


def summarize_article(
    article: Article, min_content_length: int = 300
) -> Optional[Dict]:
    logger.debug("Summarizing: %s", article.title)

    try:
        llm_client = _get_client()
        if llm_client is None:
            logger.warning("Anthropic client unavailable, skipping: %s", article.title)
            return None

        if not article.content or len(article.content) < min_content_length:
            logger.debug("Content too short, skipping: %s", article.title)
            return None

        text = article.content[:12000]
        words = len(article.content.split())
        reading_time = round(words / 200)

        response = llm_client.messages.create(
            model=MODEL_NAME,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"{USER_PROMPT_TEMPLATE}\n\n"
                        f"Title: {article.title}\n\n"
                        f"Article:\n{text}"
                    ),
                },
            ],
        )
        output_text = response.content[0].text
        parsed = json.loads(output_text)

        result = {
            "title": article.title,
            "source": article.source,
            "url": article.link,
            "reading_time": reading_time,
            "importance_score": int(parsed["importance_score"]),
            "tldr": parsed["tldr"],
            "summary": parsed["summary"],
            "key_insights": parsed["key_insights"],
        }
        return result
    except Exception as exc:
        logger.warning("Summarization failed for %s: %s", article.title, exc)
        return None

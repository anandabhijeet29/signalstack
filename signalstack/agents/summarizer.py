from typing import Dict, Optional
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from signalstack.models.article import Article
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parents[2] / ".env")
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-5.3-chat-latest")

client: Optional[OpenAI] = None


def _get_client() -> Optional[OpenAI]:
    global client
    if client is not None:
        return client

    try:
        client = OpenAI()
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

summary: A 2–3 sentence explanation of the article’s argument.

key_insights: A list of 3–5 important insights derived from the article.

importance_score: An integer from 1 to 10 representing how important the article is for an AI/technology intelligence digest.

Scoring guidelines:

1–3 → minor commentary or opinion
4–6 → moderately interesting analysis
7–8 → important development or insight
9–10 → major signal with strategic implications
"""


def summarize_article(article: Article) -> Optional[Dict]:
    print(f"LLM summarizing article: {article.title}")

    try:
        llm_client = _get_client()
        if llm_client is None:
            print(f"LLM summarization failed: {article.title}")
            return None

        if not article.content or len(article.content) < 500:
            return None

        text = article.content[:12000]
        words = len(article.content.split())
        reading_time = round(words / 200)

        response = llm_client.responses.create(
            model=MODEL_NAME,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
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
        parsed = json.loads(response.output_text)

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
        print(f"LLM summarization failed: {article.title} ({exc})")
        return None

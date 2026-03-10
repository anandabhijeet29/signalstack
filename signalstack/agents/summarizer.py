from typing import Dict, List, Optional
import re

from signalstack.models.article import Article


KEYWORDS = ("ai", "model", "agent", "compute", "research", "policy", "startup", "data")


def summarize_article(article: Article) -> Optional[Dict]:
    print(f"Summarizing: {article.title}")

    try:
        if not article.content or len(article.content) < 500:
            return None

        content = article.content[:12000]
        words = len(content.split())
        reading_time = round(words / 200)

        raw_sentences = re.split(r"(?<=[.!?]) +", content)
        sentences = [sentence.strip() for sentence in raw_sentences if sentence.strip()]
        if not sentences:
            return None

        tldr = sentences[0]
        summary = " ".join(sentences[:3]).strip()

        insights: List[str] = []
        seen = set()

        for sentence in sentences:
            if len(sentence) < 30:
                continue

            normalized = sentence.lower()
            if normalized in seen:
                continue

            if any(keyword in normalized for keyword in KEYWORDS):
                insights.append(sentence)
                seen.add(normalized)
                if len(insights) == 5:
                    break

        if len(insights) < 3:
            for sentence in sentences:
                if len(insights) == 5:
                    break
                if len(sentence) < 30:
                    continue

                normalized = sentence.lower()
                if normalized in seen:
                    continue

                insights.append(sentence)
                seen.add(normalized)

        if not insights:
            return None

        result = {
            "title": article.title,
            "source": article.source,
            "url": article.link,
            "reading_time": reading_time,
            "tldr": tldr,
            "summary": summary,
            "key_insights": insights[:5],
        }
        print(f"Summary generated for: {article.title}")
        return result
    except Exception:
        return None

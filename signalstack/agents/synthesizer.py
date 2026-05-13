from typing import List, Dict, Optional
import json
import os

from openai import OpenAI


MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o")

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
    "You are an analyst producing a weekly intelligence briefing from multiple "
    "articles. Identify the most important themes that appear across the articles."
)

USER_PROMPT = (
    "Analyze the following article summaries.\n\n"
    "Identify the 3-5 most important themes or developments that appear across "
    "these articles.\n\n"
    "Return JSON with:\n\n"
    'themes: ["theme 1", "theme 2", "theme 3"]'
)


def synthesize_themes(summaries: List[Dict]) -> Optional[List[str]]:
    if not summaries:
        return []

    llm_client = _get_client()
    if llm_client is None:
        return None

    blocks: List[str] = []
    for item in summaries:
        title = str(item.get("title", "")).strip()
        tldr = str(item.get("tldr", "")).strip()
        insights = item.get("key_insights", []) or []

        lines = [f"Title: {title}", f"TLDR: {tldr}", "Insights:"]
        for insight in insights:
            lines.append(f"- {insight}")
        blocks.append("\n".join(lines))

    combined_text = "\n\n".join(blocks)

    try:
        try:
            response = llm_client.responses.create(
                model=MODEL_NAME,
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"{USER_PROMPT}\n\n{combined_text}"},
                ],
                temperature=0.2,
            )
        except Exception:
            response = llm_client.responses.create(
                model=MODEL_NAME,
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"{USER_PROMPT}\n\n{combined_text}"},
                ],
            )

        parsed = json.loads(response.output_text)
        themes = parsed.get("themes")
        if not isinstance(themes, list):
            return None

        cleaned = [str(theme).strip() for theme in themes if str(theme).strip()]
        return cleaned or None
    except Exception:
        return None

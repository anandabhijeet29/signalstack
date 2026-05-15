from typing import List, Dict, Optional
import json
import os

from anthropic import Anthropic


MODEL_NAME = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-5")

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
        response = llm_client.messages.create(
            model=MODEL_NAME,
            max_tokens=512,
            system=SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": f"{USER_PROMPT}\n\n{combined_text}"},
            ],
        )

        output_text = response.content[0].text
        parsed = json.loads(output_text)
        themes = parsed.get("themes")
        if not isinstance(themes, list):
            return None

        cleaned = [str(theme).strip() for theme in themes if str(theme).strip()]
        return cleaned or None
    except Exception:
        return None

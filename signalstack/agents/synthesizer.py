from typing import List, Dict, Optional
import json

from openai import OpenAI


client = OpenAI()

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
    print("Synthesizing major themes across articles..")

    try:
        if not summaries:
            return []

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
            response = client.responses.create(
                model="gpt-5.3-chat-latest",
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"{USER_PROMPT}\n\n{combined_text}"},
                ],
                temperature=0.2,
            )
        except Exception:
            # Fallback for models that do not accept temperature in Responses API.
            response = client.responses.create(
                model="gpt-5.3-chat-latest",
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
        print("Theme synthesis failed")
        return None

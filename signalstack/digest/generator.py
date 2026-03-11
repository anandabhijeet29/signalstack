from typing import List, Dict
from datetime import datetime
from pathlib import Path


def generate_digest(summaries: List[Dict]) -> str:
    date_str = datetime.now().strftime("%Y-%m-%d")
    sorted_summaries = sorted(
        summaries,
        key=lambda item: int(item.get("importance_score", 0)),
        reverse=True,
    )

    lines = [
        "---",
        "title: SignalStack Weekly Intelligence",
        f"date: {date_str}",
        "tags: [signalstack, ai, intelligence]",
        "---",
        "",
        "# SignalStack Weekly Intelligence",
        "",
        f"Week of {date_str}",
        "",
        "## Top Signals",
        "",
    ]

    for summary in sorted_summaries:
        title = summary.get("title", "Untitled")
        source = summary.get("source", "Unknown")
        reading_time = summary.get("reading_time", 0)
        tldr = summary.get("tldr", "")
        body_summary = summary.get("summary", "")
        insights = summary.get("key_insights", [])
        url = summary.get("url", "")

        lines.extend(
            [
                f"### {title}",
                "",
                f"Source: {source}",
                f"Reading Time: {reading_time} min",
                "",
                "#### TLDR",
                tldr,
                "",
                "#### Summary",
                body_summary,
                "",
                "#### Key Insights",
            ]
        )

        for insight in insights:
            lines.append(f"• {insight}")

        lines.extend(
            [
                "",
                "#### Read Full Article",
                url,
                "",
                "---",
                "",
            ]
        )

    return "\n".join(lines).strip() + "\n"


def save_digest(markdown: str, vault_path: str) -> None:
    date_str = datetime.now().strftime("%Y_%m_%d")
    vault_dir = Path(vault_path)
    vault_dir.mkdir(parents=True, exist_ok=True)

    file_path = vault_dir / f"signalstack_weekly_{date_str}.md"
    file_path.write_text(markdown, encoding="utf-8")
    print(f"Digest saved to {file_path}")

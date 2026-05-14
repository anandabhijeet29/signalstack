import logging
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

import typer

from signalstack.pipeline import PipelineConfig, run_pipeline


app = typer.Typer()


@app.command()
def run(
    top_n: int = typer.Option(5, help="Number of top ranked articles to keep."),
    max_age_days: int = typer.Option(
        7, help="How many days to keep seen URLs before expiring."
    ),
    min_content_length: int = typer.Option(
        300, help="Minimum extracted content length to keep as article content."
    ),
    max_entries_per_feed: int = typer.Option(
        5, help="Maximum number of entries to fetch per RSS feed."
    ),
    vault_path: str = typer.Option(
        "", help="Path to Obsidian vault folder where digest markdown is saved."
    ),
    investigate: bool = typer.Option(
        False,
        "--investigate",
        help="Run the agentic investigator after summarization. Requires OPENAI_API_KEY.",
    ),
    max_steps: int = typer.Option(
        5,
        "--max-steps",
        help="Maximum number of tool-call steps the investigator can make.",
    ),
    max_urls: int = typer.Option(
        10,
        "--max-urls",
        help="Maximum number of URLs the investigator can fetch.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    config = PipelineConfig(
        top_n=top_n,
        max_age_days=max_age_days,
        min_content_length=min_content_length,
        max_entries_per_feed=max_entries_per_feed,
        vault_path=vault_path or None,
        investigate=investigate,
        max_steps=max_steps,
        max_urls=max_urls,
    )
    top_articles, summaries = run_pipeline(config=config)
    if not top_articles:
        return

    typer.echo(f"\nFound {len(top_articles)} top articles.")
    typer.echo(f"Generated {len(summaries)} summaries.")
    typer.echo("\nTop ranked articles:\n")
    for article in top_articles:
        typer.echo(f"- {article.title}\n  {article.link}")


if __name__ == "__main__":
    app()

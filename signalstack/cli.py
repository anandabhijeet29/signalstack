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
    vault_path: str = typer.Option(
        "", help="Path to Obsidian vault folder where digest markdown is saved."
    ),
) -> None:
    config = PipelineConfig(
        top_n=top_n,
        max_age_days=max_age_days,
        min_content_length=min_content_length,
        vault_path=vault_path or None,
    )
    top_articles, summaries = run_pipeline(config=config)
    if not top_articles:
        return

    typer.echo(f"Found {len(top_articles)} top articles.")
    typer.echo(f"Generated {len(summaries)} summaries.")
    print("\nTop ranked articles:\n")
    for article in top_articles:
        title = article.title
        url = article.link
        typer.echo(f"- {title}\n  {url}")


if __name__ == "__main__":
    app()

from pathlib import Path
import typer

from signalstack.ingestion.feed_loader import load_feeds
from signalstack.ingestion.rss_reader import fetch_articles

app = typer.Typer()


@app.command()
def run() -> None:
    feeds_path = Path(__file__).resolve().parent / "data" / "feeds.yaml"
    feed_urls = load_feeds(str(feeds_path))
    articles = fetch_articles(feed_urls)

    typer.echo(f"Found {len(articles)} articles.")
    for article in articles:
        title = article.get("title", "")
        url = article.get("link", "")
        typer.echo(f"- {title}\n  {url}")


if __name__ == "__main__":
    app()

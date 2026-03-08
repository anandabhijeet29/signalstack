from pathlib import Path
import typer

from signalstack.ingestion.feed_loader import load_feeds
from signalstack.ingestion.rss_reader import fetch_articles
from signalstack.agents.ranker import rank_articles


app = typer.Typer()


@app.command()
def run() -> None:
    feeds_path = Path(__file__).resolve().parent / "data" / "feeds.yaml"
    feed_urls = load_feeds(str(feeds_path))
    articles = fetch_articles(feed_urls)
    if not articles:
        print("No articles fetched. Check network or feed availability.")
        return
    if len(articles) < 3:
        print("Warning: very few articles fetched. Network or feed issue likely.")

    top_n = 5
    typer.echo(f"[DEBUG] Ranking {len(articles)} articles with top_n={top_n}")
    top_articles = rank_articles(articles, top_n=top_n)
    typer.echo(f"[DEBUG] Ranked result count: {len(top_articles)}")

    typer.echo(f"Found {len(top_articles)} top articles.")
    print("\nTop ranked articles:\n")
    for article in top_articles:
        title = article.get("title", "")
        url = article.get("link", "")
        typer.echo(f"- {title}\n  {url}")


if __name__ == "__main__":
    app()

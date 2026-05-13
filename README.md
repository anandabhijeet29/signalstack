# SignalStack

SignalStack is an AI-powered intelligence agent that reads tech and AI newsletters via RSS, extracts high-signal insights using LLM analysis, and generates a weekly intelligence digest as Obsidian-compatible markdown.

## How It Works

SignalStack runs a multi-stage pipeline:

1. **Ingest** -- Fetches the latest articles from RSS feeds defined in `signalstack/data/feeds.yaml`
2. **Deduplicate** -- Filters out previously seen articles using a local JSON store with configurable TTL
3. **Rank** -- Scores articles by keyword relevance (AI, compute, policy, startups, etc.) and title quality
4. **Extract** -- Pulls full article text using [trafilatura](https://github.com/adbar/trafilatura), then re-ranks with richer content
5. **Summarize** -- Sends each article to an OpenAI model to produce a structured summary (TLDR, key insights, importance score 1-10)
6. **Synthesize** -- Identifies 3-5 cross-article themes from the combined summaries
7. **Publish** -- Generates a markdown digest sorted by importance and saves it to an Obsidian vault

## Default Feeds

- [Stratechery](https://stratechery.com)
- [Noahpinion](https://www.noahpinion.blog)
- [Not Boring](https://www.notboring.co)
- [Latent Space](https://www.latent.space)
- [The Zvi](https://thezvi.substack.com)

Add or remove feeds in [`signalstack/data/feeds.yaml`](signalstack/data/feeds.yaml).

## Setup

```bash
# Clone the repo
git clone https://github.com/yourusername/signalstack.git
cd signalstack

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install typer openai feedparser requests pyyaml python-dotenv rich

# Configure your OpenAI API key
echo "OPENAI_API_KEY=sk-..." > .env
```

Optionally set `OPENAI_MODEL` in `.env` to override the default model.

## Usage

```bash
python -m signalstack.cli run
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--top-n` | `5` | Number of top articles to include in the digest |
| `--max-age-days` | `7` | Days before a seen URL expires and can reappear |
| `--min-content-length` | `300` | Minimum extracted text length (characters) to use an article |
| `--vault-path` | `""` | Path to Obsidian vault folder for saving the digest |

### Example

```bash
python -m signalstack.cli run --top-n 10 --vault-path ~/my-vault/Intelligence
```

## Output

The digest is saved as `signalstack_weekly_YYYY_MM_DD.md` with YAML frontmatter, and includes:

- Major cross-article themes
- Per-article sections with TLDR, summary, key insights, importance score, reading time, and source link

## Tech Stack

- **Language**: Python 3
- **CLI**: [Typer](https://typer.tiangolo.com/)
- **LLM**: [OpenAI API](https://platform.openai.com/) (Responses API)
- **RSS Parsing**: [feedparser](https://github.com/kurtmckee/feedparser)
- **Text Extraction**: [trafilatura](https://github.com/adbar/trafilatura)
- **Config**: PyYAML, python-dotenv
- **Output**: Markdown with YAML frontmatter (Obsidian-compatible)

## Project Structure

```
signalstack/
  cli.py                  # Typer CLI entry point
  pipeline.py             # Pipeline orchestration
  models/
    article.py            # Article dataclass
  ingestion/
    feed_loader.py        # Load feed URLs from YAML
    rss_reader.py         # Fetch and parse RSS feeds
  processing/
    extractor.py          # Full-text extraction via trafilatura
    article_store.py      # Seen-article tracking (JSON)
  agents/
    ranker.py             # Keyword + title scoring
    summarizer.py         # LLM-powered article summarization
    synthesizer.py        # Cross-article theme synthesis
  digest/
    generator.py          # Markdown digest generation
  data/
    feeds.yaml            # RSS feed URLs
    seen_articles.json    # Auto-generated seen URL store
```

## License

MIT

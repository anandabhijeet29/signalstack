# SignalStack

An AI-powered intelligence agent that reads tech and AI newsletters via RSS, autonomously investigates the most interesting claims, and generates a weekly digest with a visible investigation log showing every research decision.

## What It Does

Most newsletter summarizers stop at the summary. SignalStack goes further: after summarizing articles, an agentic investigator reads the summaries, decides which claims are worth verifying, uses tools to fetch evidence, and writes an investigation log explaining what it found and why it matters.

The investigation log is the differentiator. It shows the agent's reasoning — not just what it summarized, but what it chose to investigate, what it found, and how different articles connect or contradict each other.

**Sample — from `examples/sample_investigation.md`:**

```
### Thread: "Compute scaling hitting an efficiency wall"

**Trigger:** Articles #2 and #4 both describe diminishing returns on raw compute
scaling — but reach opposite conclusions. The contradiction is worth resolving.

**Decision:** Search for recent research on whether efficiency gains from architecture
improvements are outpacing raw compute scaling.

**Action:** Searched for "compute scaling efficiency wall 2026 architecture improvements"

**Found:** DeepMind paper argues MoE architectures achieve GPT-4 performance at 40%
compute cost. Anthropic research shows inference-time compute is the new scaling lever.

**Key finding:** The "wall" narrative is technically correct about training compute
but misleading — inference-time compute and architectural efficiency are filling the gap.

**Connection:** This reframes Article #1's economic impact thesis. Inference-time
compute is deployable immediately — impact arrives faster than training timelines suggest.
```

See [`examples/sample_investigation.md`](examples/sample_investigation.md) and [`examples/sample_digest.md`](examples/sample_digest.md) for full examples.

## How It Works

SignalStack runs a multi-stage pipeline:

1. **Ingest** — Fetches articles from RSS feeds concurrently using a thread pool
2. **Deduplicate** — Filters previously seen articles using a local JSON store with configurable TTL
3. **Rank** — Scores articles by keyword relevance (AI, compute, policy, startups) and title quality
4. **Extract** — Pulls full article text concurrently via [trafilatura](https://github.com/adbar/trafilatura), then re-ranks with richer content
5. **Summarize** — Sends each article to an OpenAI model for structured summary (TLDR, key insights, importance score 1–10)
6. **Synthesize** — Identifies 3–5 cross-article themes from the combined summaries
7. **Investigate** *(optional)* — Agentic loop that autonomously investigates interesting threads using a tool registry (`search_web`, `fetch_and_extract`, `find_related`), writes a visible investigation log
8. **Publish** — Generates a markdown digest sorted by importance, saved to an Obsidian vault or printed to stdout

## Architecture

The investigator uses raw OpenAI tool-use calls with a hand-rolled tool registry — no agent framework. The agent loop runs until the LLM concludes or the research budget is exhausted.

```
RSS → Rank → Extract → Summarize → Synthesize → Digest
                                        ↓
                               Investigator Agent
                                 ├── Tool Registry
                                 │   ├── search_web (Tavily)
                                 │   ├── fetch_and_extract (trafilatura)
                                 │   └── find_related (cosine similarity)
                                 ├── Agent Loop (budget: 5 steps / 10 URLs)
                                 └── Investigation Log → Digest
```

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
git clone https://github.com/anandabhijeet29/signalstack.git
cd signalstack

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install the package
pip install -e .

# Configure API keys
echo "OPENAI_API_KEY=sk-..." > .env

# Optional: enable web search in the investigator
echo "TAVILY_API_KEY=tvly-..." >> .env
```

Optionally set `OPENAI_MODEL` in `.env` to override the default model (`gpt-4o`).

A free [Tavily](https://tavily.com) API key (1,000 searches/month) enables `search_web`. Without it, the investigator still works using `fetch_and_extract` and `find_related`.

## Usage

```bash
# Standard run — summarize and generate digest
signalstack run

# With agentic investigation
signalstack run --investigate

# Investigation with custom budget
signalstack run --investigate --max-steps 10 --max-urls 15

# Save to Obsidian vault
signalstack run --investigate --vault-path ~/my-vault/Intelligence
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--top-n` | `5` | Number of top articles to include in the digest |
| `--max-age-days` | `7` | Days before a seen URL expires and can reappear |
| `--min-content-length` | `300` | Minimum extracted text length (chars) to use an article |
| `--max-entries-per-feed` | `5` | Maximum entries to fetch per RSS feed |
| `--vault-path` | `""` | Path to Obsidian vault folder for saving the digest |
| `--investigate` | `false` | Run the agentic investigator after summarization |
| `--max-steps` | `5` | Maximum tool-call steps for the investigator |
| `--max-urls` | `10` | Maximum URLs the investigator can fetch |
| `--verbose` / `-v` | `false` | Enable debug logging |

When `--vault-path` is omitted, the digest is printed to stdout.

## Output

The digest is saved as `signalstack_weekly_YYYY_MM_DD.md` with YAML frontmatter. With `--investigate`, it includes:

- Major cross-article themes
- **Investigation log** — agent reasoning, tool calls, findings, cross-article connections
- Per-article sections with TLDR, summary, key insights, importance score, reading time, and source link

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.10+ |
| CLI | [Typer](https://typer.tiangolo.com/) |
| LLM / Agent | [OpenAI API](https://platform.openai.com/) (Responses API, tool-use) |
| RSS Parsing | [feedparser](https://github.com/kurtmckee/feedparser) |
| Text Extraction | [trafilatura](https://github.com/adbar/trafilatura) |
| Web Search | [Tavily](https://tavily.com) (optional) |
| Cosine Similarity | Python stdlib (`collections.Counter`) |
| Config | PyYAML, python-dotenv |
| Testing | pytest |
| Output | Markdown with YAML frontmatter (Obsidian-compatible) |

## Testing

```bash
pip install pytest
python -m pytest tests/ -v
```

The test suite covers all pipeline stages with mocked LLM calls, including the agent loop (budget exhaustion, tool dispatch, consecutive failure handling) and tool implementations.

## Project Structure

```
signalstack/
  cli.py                  # Typer CLI entry point
  pipeline.py             # Pipeline orchestration
  models/
    article.py            # Article dataclass
  ingestion/
    feed_loader.py        # Load feed URLs from YAML
    rss_reader.py         # Concurrent RSS feed fetching
  processing/
    extractor.py          # Concurrent full-text extraction via trafilatura
    article_store.py      # Seen-article tracking (JSON)
  agents/
    ranker.py             # Keyword + title scoring
    summarizer.py         # LLM-powered article summarization
    synthesizer.py        # Cross-article theme synthesis
    investigator.py       # Agentic research loop (tool registry + budget)
    tools.py              # Tool implementations (search, fetch, find_related)
    trace.py              # Investigation trace and log formatting
  digest/
    generator.py          # Markdown digest generation
  data/
    feeds.yaml            # RSS feed URLs
examples/
  sample_investigation.md # Sample investigation log
  sample_digest.md        # Sample full digest with investigation
tests/
  test_ranker.py          # Ranking and scoring tests
  test_article_store.py   # Deduplication and persistence tests
  test_extractor.py       # Text extraction tests
  test_feed_loader.py     # YAML feed loading tests
  test_generator.py       # Digest generation tests
  test_summarizer.py      # LLM summarization tests (mocked)
  test_synthesizer.py     # Theme synthesis tests (mocked)
  test_investigator.py    # Agent loop tests (mocked)
  test_tools.py           # Tool implementation tests
```

## License

MIT

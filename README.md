# SignalStack

An AI-powered intelligence pipeline that reads tech and AI newsletters via RSS, autonomously investigates the most interesting claims, and generates a weekly digest — optionally including a voice debate between two AI personas arguing over the week's biggest stories.

## What It Does

Most newsletter summarizers stop at the summary. SignalStack has three layers beyond that:

1. **Investigation** — An agentic Sonnet-powered loop reads the summaries, picks 2–3 threads worth verifying, uses tools to fetch evidence, and writes a visible investigation log explaining what it found and why it matters.

2. **Debate** — A Haiku-powered text debate between a Skeptic and an Optimist, each primed with asymmetric evidence from the investigation log. The skeptic gets contradictions; the optimist gets breakthroughs.

3. **Voice** — The debate can be rendered as audio using ElevenLabs: either via ConvAI agents (two live WebSocket sessions) or via the TTS fallback (Anthropic generates text, ElevenLabs synthesizes each turn with a distinct voice).

The investigation log is the core differentiator — it shows the agent's reasoning, not just what it summarized.

**Sample investigation thread:**

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

## How It Works

```
RSS feeds
  → Rank (keyword + title scoring)
  → Extract (trafilatura, concurrent)
  → Re-rank (with full article text)
  → Summarize (Claude Haiku, structured JSON)
  → Synthesize themes (Claude Haiku)
  → Investigate (Claude Sonnet, tool-use loop)   ← optional
  → Debate (Claude Haiku, asymmetric context)     ← optional
  → Digest (Markdown, Obsidian-compatible)
        + investigation_YYYY_MM_DD.json sidecar

signalstack debate --from digest.md
  → Load sidecar JSON (summaries + investigation trace)
  → Build debate scaffold (Claude Haiku)
  → Build asymmetric context (skeptic ← contradictions, optimist ← breakthroughs)
  → Voice debate (ElevenLabs ConvAI or TTS fallback)
  → MP3 / WAV output
```

### Investigator agent

The investigator runs a raw Anthropic tool-use loop — no agent framework. It continues until it reaches a natural conclusion (`stop_reason=end_turn`) or exhausts its research budget. Every step is recorded in an `InvestigationTrace` and rendered as a formatted markdown log.

Tool registry:
- `search_web` — Tavily search (requires `TAVILY_API_KEY`; excluded if not set)
- `fetch_and_extract` — Full-text extraction via trafilatura
- `find_related` — Cosine similarity over article summaries

### Debate agent

Two debate paths share the same scaffold and asymmetric context:

- **ConvAI** (default) — Two ElevenLabs `Conversation` sessions, one per persona. `send_user_message()` injects the previous speaker's text. System prompts and first-message suppression are applied via `override_agent_config`.
- **TTS fallback** (`--no-conversational-ai`) — Anthropic generates each turn as a stateless call; ElevenLabs TTS synthesizes audio with a distinct voice per persona. Reliable for environments without ConvAI agents configured.

Each debate has an intro narration, N alternating turns, and closing statements from both personas.

## Default Feeds

- [Stratechery](https://stratechery.com)
- [Noahpinion](https://www.noahpinion.blog)
- [Not Boring](https://www.notboring.co)
- [Latent Space](https://www.latent.space)
- [The Zvi](https://thezvi.substack.com)

Add or remove feeds in [`signalstack/data/feeds.yaml`](signalstack/data/feeds.yaml).

## Setup

```bash
git clone https://github.com/anandabhijeet29/signalstack.git
cd signalstack

python -m venv .venv
source .venv/bin/activate

pip install -e .
```

Create a `.env` file:

```bash
# Required
ANTHROPIC_API_KEY=sk-ant-...

# Optional: override models (defaults shown)
ANTHROPIC_MODEL=claude-haiku-4-5-20251001          # summarizer, synthesizer, debate text
ANTHROPIC_INVESTIGATOR_MODEL=claude-sonnet-4-6     # investigator agent loop

# Optional: enable web search in the investigator
TAVILY_API_KEY=tvly-...

# Required for voice debate
ELEVENLABS_API_KEY=sk_...

# Required for ConvAI debate (create agents at elevenlabs.io/app/conversational-ai)
ELEVENLABS_SKEPTIC_AGENT_ID=agent_...
ELEVENLABS_OPTIMIST_AGENT_ID=agent_...

# Optional: override TTS voices for the fallback path (defaults from debate_personas.yaml)
ELEVENLABS_SKEPTIC_VOICE_ID=EXAVITQu4vr4xnSDxMaL
ELEVENLABS_OPTIMIST_VOICE_ID=TX3LPaxmHKxFdv7VOQHJ
```

A free [Tavily](https://tavily.com) key (1,000 searches/month) enables `search_web`. Without it, the investigator still works using `fetch_and_extract` and `find_related`.

For voice output, [ffmpeg](https://ffmpeg.org/download.html) enables MP3 export. Without it, the audio saves as WAV automatically.

## Usage

### `signalstack run`

```bash
# Summarize and generate digest
signalstack run

# With investigation log
signalstack run --investigate

# With investigation + text debate
signalstack run --investigate --debate

# Save to Obsidian vault (also writes investigation_YYYY_MM_DD.json sidecar)
signalstack run --investigate --vault-path ~/vault/Intelligence

# Tune the investigator budget
signalstack run --investigate --max-steps 10 --max-urls 15

# Debug logging
signalstack run --investigate --debate --verbose
```

| Flag | Default | Description |
|------|---------|-------------|
| `--top-n` | `5` | Articles to include in the digest |
| `--max-age-days` | `7` | Days before a seen URL expires |
| `--min-content-length` | `300` | Minimum extracted text length (chars) |
| `--max-entries-per-feed` | `5` | Max entries per RSS feed |
| `--vault-path` | `""` | Obsidian vault folder; prints to stdout if omitted |
| `--investigate` | `false` | Run the agentic investigator |
| `--max-steps` | `5` | Max tool-call steps for the investigator |
| `--max-urls` | `10` | Max URLs the investigator can fetch |
| `--debate` | `false` | Run a text debate and include it in the digest |
| `--debate-rounds` | `3` | Rounds of back-and-forth (skeptic + optimist per round) |
| `--verbose` / `-v` | `false` | Debug logging |

### `signalstack debate`

Loads a previously saved digest and its investigation sidecar, then runs a voice debate.

```bash
# Run first to generate the sidecar JSON
signalstack run --investigate --vault-path ~/vault/Intelligence

# Then run the voice debate (ConvAI — requires ELEVENLABS_*_AGENT_ID)
signalstack debate --from ~/vault/Intelligence/signalstack_weekly_2026_05_16.md

# TTS fallback (requires only ELEVENLABS_API_KEY, no ConvAI agents)
signalstack debate --from ~/vault/Intelligence/signalstack_weekly_2026_05_16.md \
  --no-conversational-ai

# Custom personas and turn count
signalstack debate --from digest.md --personas doomer,accelerationist --max-turns 8
```

| Flag | Default | Description |
|------|---------|-------------|
| `--from` | `""` | Path to digest `.md`; finds `investigation_YYYY_MM_DD.json` alongside it |
| `--personas` | `skeptic,optimist` | Two persona names from `debate_personas.yaml` |
| `--max-turns` | `6` | Total debate turns (excluding intro and closings) |
| `--no-conversational-ai` | `false` | Use TTS fallback instead of ElevenLabs ConvAI |
| `--output` | `debate_YYYY-MM-DD.mp3` | Output audio file path |
| `--verbose` / `-v` | `false` | Debug logging |

Available personas: `skeptic`, `optimist`, `doomer`, `accelerationist`. Add custom personas in [`signalstack/data/debate_personas.yaml`](signalstack/data/debate_personas.yaml).

## Output

`signalstack run --investigate --debate --vault-path ~/vault` produces:

- `signalstack_weekly_YYYY_MM_DD.md` — Full digest with:
  - Major cross-article themes
  - Investigation log (agent reasoning, tool calls, findings, connections)
  - Debate transcript (intro → turns → closing statements)
  - Per-article sections: TLDR, summary, key insights, importance score, reading time, link
- `investigation_YYYY_MM_DD.json` — Sidecar with summaries + investigation trace for `signalstack debate --from`

`signalstack debate` produces:
- `debate_YYYY-MM-DD.mp3` (or `.wav` if ffmpeg is not installed)
- Transcript printed to stdout

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.10+ |
| CLI | [Typer](https://typer.tiangolo.com/) |
| LLM / Agents | [Anthropic API](https://docs.anthropic.com/) — Haiku (summarizer, synthesizer, debate) + Sonnet (investigator) |
| RSS Parsing | [feedparser](https://github.com/kurtmckee/feedparser) |
| Text Extraction | [trafilatura](https://github.com/adbar/trafilatura) |
| Web Search | [Tavily](https://tavily.com) (optional) |
| Voice Synthesis | [ElevenLabs](https://elevenlabs.io) — ConvAI or TTS |
| Audio Output | [pydub](https://github.com/jiaaro/pydub) + [sounddevice](https://python-sounddevice.readthedocs.io/) |
| Config | PyYAML, python-dotenv |
| Testing | pytest |
| Output | Markdown with YAML frontmatter (Obsidian-compatible) |

## Project Structure

```
signalstack/
  cli.py                    # Typer CLI — 'run' and 'debate' commands
  pipeline.py               # Pipeline orchestration + investigation sidecar
  models/
    article.py              # Article dataclass
  ingestion/
    feed_loader.py          # Load feed URLs from YAML
    rss_reader.py           # Concurrent RSS feed fetching
  processing/
    extractor.py            # Concurrent full-text extraction via trafilatura
    article_store.py        # Seen-article tracking (JSON, TTL-based)
  agents/
    ranker.py               # Keyword + title scoring
    summarizer.py           # Claude Haiku article summarization
    synthesizer.py          # Cross-article theme synthesis
    investigator.py         # Agentic research loop (Sonnet + tool registry + budget)
    tools.py                # Tool implementations: search_web, fetch_and_extract, find_related
    trace.py                # InvestigationTrace — step recording, JSON serialization, markdown log
    debate_context.py       # Debate scaffold generation + asymmetric system prompts
    debate_agent.py         # DebateOrchestrator (ConvAI + TTS fallback) + run_text_debate
  digest/
    generator.py            # Markdown digest assembly
  data/
    feeds.yaml              # RSS feed URLs
    debate_personas.yaml    # Persona configs: name, description, voice_id
tests/
  ...                       # pytest suite covering all pipeline stages
```

## License

MIT

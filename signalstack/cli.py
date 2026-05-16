import json
import logging
import os
import re
from datetime import date
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
        help="Run the agentic investigator after summarization. Requires ANTHROPIC_API_KEY.",
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
    debate: bool = typer.Option(
        False,
        "--debate",
        help="Run a text debate between skeptic/optimist personas and include in digest.",
    ),
    debate_rounds: int = typer.Option(
        3,
        "--debate-rounds",
        help="Number of debate rounds (one skeptic + one optimist turn per round).",
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
        debate=debate,
        debate_rounds=debate_rounds,
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


@app.command()
def debate(
    from_digest: str = typer.Option(
        "",
        "--from",
        help=(
            "Path to a saved digest .md file. Looks for investigation_YYYY_MM_DD.json "
            "in the same directory. Run 'signalstack run --investigate' first."
        ),
    ),
    personas: str = typer.Option(
        "skeptic,optimist",
        help="Comma-separated persona names from debate_personas.yaml.",
    ),
    max_turns: int = typer.Option(6, "--max-turns", help="Maximum total debate turns."),
    output: str = typer.Option(
        "",
        help="Path to save debate MP3. Defaults to ./debate_YYYY_MM_DD.mp3",
    ),
    no_conversational_ai: bool = typer.Option(
        False,
        "--no-conversational-ai",
        help=(
            "Use TTS fallback instead of ElevenAgents Conversation. "
            "Anthropic generates turn text, ElevenLabs synthesizes audio."
        ),
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    """Run a live AI voice debate from a SignalStack investigation.

    Requires: ELEVENLABS_API_KEY, ELEVENLABS_SKEPTIC_AGENT_ID, ELEVENLABS_OPTIMIST_AGENT_ID
    and (for debate scaffold generation) ANTHROPIC_API_KEY.

    Example:
        signalstack run --investigate --vault-path ~/vault
        signalstack debate --from ~/vault/signalstack_weekly_2026_05_16.md
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    # --- Validate environment ---
    api_key = os.environ.get("ELEVENLABS_API_KEY", "")
    if not api_key:
        typer.echo("Error: ELEVENLABS_API_KEY is required. Add it to .env", err=True)
        raise typer.Exit(1)

    skeptic_agent_id = os.environ.get("ELEVENLABS_SKEPTIC_AGENT_ID", "")
    optimist_agent_id = os.environ.get("ELEVENLABS_OPTIMIST_AGENT_ID", "")
    if not no_conversational_ai and (not skeptic_agent_id or not optimist_agent_id):
        typer.echo(
            "Error: ELEVENLABS_SKEPTIC_AGENT_ID and ELEVENLABS_OPTIMIST_AGENT_ID are required.\n"
            "Create two agents at https://elevenlabs.io/app/conversational-ai, "
            "then add their IDs to .env\n"
            "Or use --no-conversational-ai to run the TTS fallback.",
            err=True,
        )
        raise typer.Exit(1)

    # --- Load investigation trace + summaries ---
    from signalstack.agents.debate_context import (
        AsymmetricContext,
        build_asymmetric_context,
        build_scaffold,
        load_persona,
    )
    from signalstack.agents.debate_agent import DebateBudget, DebateOrchestrator
    from signalstack.agents.trace import InvestigationTrace

    trace: InvestigationTrace | None = None
    summaries: list = []

    if from_digest:
        digest_path = Path(from_digest).expanduser().resolve()
        if not digest_path.exists():
            typer.echo(f"Error: digest file not found: {digest_path}", err=True)
            raise typer.Exit(1)

        # Find investigation_YYYY_MM_DD.json in same directory
        investigation_path = _find_investigation_json(digest_path)
        typer.echo(f"Loading investigation from: {investigation_path}")
        with open(investigation_path) as f:
            inv_data = json.load(f)
        trace = InvestigationTrace.from_dict(inv_data)
        summaries = inv_data.get("summaries", [])
    else:
        typer.echo(
            "No --from path provided. Generating debate scaffold from scratch "
            "(no investigation context)."
        )

    # --- Load personas ---
    persona_names = [p.strip() for p in personas.split(",")]
    if len(persona_names) != 2:
        typer.echo("Error: --personas requires exactly two comma-separated names.", err=True)
        raise typer.Exit(1)

    try:
        skeptic_persona = load_persona(persona_names[0])
        optimist_persona = load_persona(persona_names[1])
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    # --- Build debate scaffold and asymmetric context ---
    typer.echo("Generating debate scaffold...")
    scaffold = build_scaffold(trace, summaries)
    typer.echo(f"Debate: {scaffold.get('title', 'AI Weekly Debate')}")
    typer.echo(f"Topics: {len(scaffold.get('topics', []))}")

    context = build_asymmetric_context(trace, summaries, skeptic_persona, optimist_persona)

    # --- Run debate ---
    output_path = output or f"debate_{date.today().isoformat()}.mp3"
    budget = DebateBudget(max_turns=max_turns)
    orchestrator = DebateOrchestrator(
        elevenlabs_api_key=api_key,
        skeptic_agent_id=skeptic_agent_id,
        optimist_agent_id=optimist_agent_id,
        budget=budget,
        use_conversational_ai=not no_conversational_ai,
    )

    mode = "ElevenAgents ConvAI" if not no_conversational_ai else "TTS fallback"
    typer.echo(f"\nStarting debate ({max_turns} turns, {mode})...\n")

    turns = orchestrator.run_debate(scaffold, context)

    # --- Print transcript ---
    typer.echo("\n--- DEBATE TRANSCRIPT ---\n")
    for t in turns:
        if t.speaker == "skeptic":
            speaker_label = skeptic_persona["name"]
        elif t.speaker == "optimist":
            speaker_label = optimist_persona["name"]
        else:
            speaker_label = t.speaker.capitalize()
        typer.echo(f"{speaker_label}: {t.text}\n")

    # --- Save MP3 ---
    saved = orchestrator.save_mp3(output_path)
    if saved:
        typer.echo(f"\nSaved to: {saved}")
    else:
        typer.echo("\nNo audio captured — check ElevenLabs API key and agent IDs.")


def _find_investigation_json(digest_path: Path) -> Path:
    """Find investigation_YYYY_MM_DD.json in the same directory as the digest.

    Digest filename convention: signalstack_weekly_YYYY_MM_DD.md
    Investigation filename convention: investigation_YYYY_MM_DD.json

    Raises:
        ValueError: if date can't be extracted from filename.
        FileNotFoundError: if investigation file doesn't exist.
    """
    name = digest_path.stem  # e.g. "signalstack_weekly_2026_05_16"
    match = re.search(r"(\d{4}_\d{2}_\d{2})$", name)
    if not match:
        raise ValueError(
            f"Can't extract date from digest filename: {digest_path.name}\n"
            "Expected format: signalstack_weekly_YYYY_MM_DD.md"
        )
    date_str = match.group(1)
    investigation_path = digest_path.parent / f"investigation_{date_str}.json"
    if not investigation_path.exists():
        raise FileNotFoundError(
            f"Investigation file not found: {investigation_path}\n"
            "Run 'signalstack run --investigate' to generate it."
        )
    return investigation_path


if __name__ == "__main__":
    app()

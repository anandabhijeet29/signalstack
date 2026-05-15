"""DebateContextBuilder: takes InvestigationTrace + article summaries and produces:
  1. A debate scaffold (JSON) — the topics and angles for the debate
  2. Asymmetric system prompts — each agent gets different evidence from the same source

The asymmetry is what makes the debate interesting: the skeptic is primed with
contradictions and weak evidence; the optimist is primed with breakthroughs and
strong signals. Same investigation, different framing.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from anthropic import Anthropic

from signalstack.agents.trace import InvestigationTrace

logger = logging.getLogger(__name__)

MODEL_NAME = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-5")

# Fallback scaffold used when the LLM call fails or returns bad JSON.
_DEFAULT_SCAFFOLD: Dict[str, Any] = {
    "title": "AI Weekly: The Debate",
    "topics": [
        {
            "id": 1,
            "claim": "This week's most contested AI claim",
            "articles": [],
            "skeptic_angle": "Challenge this claim with evidence and demand specifics.",
            "optimist_angle": "Steelman this claim and argue for bigger implications.",
            "rounds": 2,
        }
    ],
}


@dataclass
class AsymmetricContext:
    """Per-agent system prompts with asymmetric evidence from the investigation log."""

    skeptic_system_prompt: str
    optimist_system_prompt: str


def load_persona(name: str) -> Dict[str, Any]:
    """Load a persona config by name from debate_personas.yaml.

    Raises:
        FileNotFoundError: if debate_personas.yaml doesn't exist.
        ValueError: if persona name isn't in the file.
    """
    yaml_path = Path(__file__).resolve().parent.parent / "data" / "debate_personas.yaml"
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    personas: Dict[str, Any] = data.get("personas", {})
    if name not in personas:
        available = list(personas.keys())
        raise ValueError(
            f"Unknown persona '{name}'. Available: {available}. "
            f"Add custom personas to signalstack/data/debate_personas.yaml."
        )
    return personas[name]


def build_scaffold(
    trace: Optional[InvestigationTrace],
    summaries: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Call OpenAI to generate a debate scaffold from the investigation log + summaries.

    The scaffold identifies the most contested topics and gives each agent an angle.
    Falls back to _DEFAULT_SCAFFOLD if the LLM call fails or returns malformed JSON.

    Args:
        trace: The investigation trace from the agentic investigator. May be None if
            the investigator hasn't run (falls back to summaries only).
        summaries: List of article summary dicts with at minimum 'title' and 'tldr' keys.

    Returns:
        Scaffold dict with 'title' and 'topics' keys. Never raises.
    """
    client = _get_anthropic_client()
    if client is None:
        logger.warning("No Anthropic client — using default scaffold")
        return _DEFAULT_SCAFFOLD

    # Build context for the LLM prompt
    context_parts: List[str] = []
    if trace and trace.conclusion:
        context_parts.append(f"INVESTIGATION LOG:\n{trace.conclusion}")
    for s in summaries[:5]:  # cap at 5 articles to keep prompt manageable
        title = str(s.get("title", "")).strip()
        tldr = str(s.get("tldr", "")).strip()
        if title:
            context_parts.append(f"Article: {title}\nTLDR: {tldr}")

    if not context_parts:
        logger.warning("No context available for scaffold — using default")
        return _DEFAULT_SCAFFOLD

    context = "\n\n".join(context_parts)

    prompt = (
        "You generate debate scaffolds from AI newsletter investigations.\n\n"
        f"{context}\n\n"
        "Identify 2-4 of the most interesting/contested claims from the above. "
        "For each, write an angle the skeptic would take and an angle the optimist would take.\n\n"
        "Return ONLY valid JSON with this exact structure:\n"
        '{"title": "short episode title", "topics": ['
        '{"id": 1, "claim": "...", "articles": ["..."], '
        '"skeptic_angle": "...", "optimist_angle": "...", "rounds": 2}'
        "]}"
    )

    try:
        response = client.messages.create(
            model=MODEL_NAME,
            max_tokens=2048,
            system="You generate structured debate scaffolds. Return only valid JSON.",
            messages=[
                {"role": "user", "content": prompt},
            ],
        )
        raw = response.content[0].text
        scaffold = json.loads(raw)
        if "topics" not in scaffold or not scaffold["topics"]:
            logger.warning("Scaffold missing 'topics' key — using default")
            return _DEFAULT_SCAFFOLD
        return scaffold
    except json.JSONDecodeError as e:
        logger.warning("Scaffold JSON parse failed: %s — using default", e)
        return _DEFAULT_SCAFFOLD
    except Exception as e:
        logger.warning("Scaffold generation failed: %s — using default", e)
        return _DEFAULT_SCAFFOLD


def build_asymmetric_context(
    trace: Optional[InvestigationTrace],
    summaries: List[Dict[str, Any]],
    skeptic_persona: Dict[str, Any],
    optimist_persona: Dict[str, Any],
) -> AsymmetricContext:
    """Build asymmetric system prompts for the two agents.

    The skeptic gets contradictions and weak evidence from the investigation log.
    The optimist gets breakthroughs and strong signals.
    Both get the same article summaries for shared factual grounding.

    Args:
        trace: Investigation trace. May be None.
        summaries: Article summaries list.
        skeptic_persona: Persona config dict (from load_persona).
        optimist_persona: Persona config dict (from load_persona).

    Returns:
        AsymmetricContext with per-agent system prompts.
    """
    # Extract contradictions and breakthroughs from trace steps
    contradictions: List[str] = []
    breakthroughs: List[str] = []

    if trace:
        for step in trace.steps:
            if not step.success or not step.result:
                continue
            text = str(step.result).lower()
            is_contradiction = any(
                w in text
                for w in ("contradict", "misleading", "wrong", "incorrect", "weak", "false", "overstated")
            )
            is_breakthrough = any(
                w in text
                for w in ("breakthrough", "significant", "major", "confirm", "strong", "real", "verified")
            )
            if is_contradiction:
                contradictions.append(str(step.result)[:300])
            elif is_breakthrough:
                breakthroughs.append(str(step.result)[:300])

    # Article summaries as shared grounding for both agents
    article_lines = [
        f"- {s.get('title', '(untitled)')}: {s.get('tldr', '')}"
        for s in summaries[:5]
    ]
    article_context = "\n".join(article_lines) if article_lines else "No articles available."

    # Skeptic evidence: contradictions first, generic challenge if none found
    if contradictions:
        skeptic_evidence = "\n".join(f"- {c}" for c in contradictions[:3])
    else:
        skeptic_evidence = (
            "No specific contradictions identified — challenge the general framing "
            "and demand sharper evidence for the strongest claims."
        )

    # Optimist evidence: breakthroughs first, generic steelman if none found
    if breakthroughs:
        optimist_evidence = "\n".join(f"- {b}" for b in breakthroughs[:3])
    else:
        optimist_evidence = (
            "No specific breakthroughs identified — steelman the most interesting "
            "claims and argue for their broader implications."
        )

    skeptic_name = skeptic_persona.get("name", "The Skeptic")
    skeptic_desc = skeptic_persona.get("description", "A skeptical debate participant.")
    optimist_name = optimist_persona.get("name", "The Optimist")
    optimist_desc = optimist_persona.get("description", "An optimistic debate participant.")

    skeptic_system_prompt = f"""You are {skeptic_name}: {skeptic_desc}

THIS WEEK'S AI NEWS (shared with your opponent):
{article_context}

YOUR PRIVATE EVIDENCE (emphasize this in the debate — your opponent doesn't have this):
{skeptic_evidence}

DEBATE RULES:
- Keep each response to 3-5 punchy sentences. Never longer.
- Engage directly with what your opponent just said before making your point.
- Cite specific evidence or findings when you push back.
- Stay in character. Never break character or acknowledge you're an AI.
- When you're done speaking, end your turn naturally — don't say "over" or "your turn"."""

    optimist_system_prompt = f"""You are {optimist_name}: {optimist_desc}

THIS WEEK'S AI NEWS (shared with your opponent):
{article_context}

YOUR PRIVATE EVIDENCE (emphasize this in the debate — your opponent doesn't have this):
{optimist_evidence}

DEBATE RULES:
- Keep each response to 3-5 punchy sentences. Never longer.
- Engage directly with what your opponent just said before making your point.
- Cite specific evidence or findings when you push further.
- Stay in character. Never break character or acknowledge you're an AI.
- When you're done speaking, end your turn naturally — don't say "over" or "your turn"."""

    return AsymmetricContext(
        skeptic_system_prompt=skeptic_system_prompt,
        optimist_system_prompt=optimist_system_prompt,
    )


# --- Anthropic client singleton ---

_anthropic_client: Optional[Anthropic] = None


def _get_anthropic_client() -> Optional[Anthropic]:
    global _anthropic_client
    if _anthropic_client is not None:
        return _anthropic_client
    try:
        _anthropic_client = Anthropic()
    except Exception as e:
        logger.warning("Failed to create Anthropic client: %s", e)
        _anthropic_client = None
    return _anthropic_client

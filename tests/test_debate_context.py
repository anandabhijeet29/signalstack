"""Tests for signalstack.agents.debate_context.

Covers:
- build_scaffold: valid trace, empty trace, LLM failure, malformed JSON
- build_asymmetric_context: asymmetric evidence extraction, persona names in prompts
- load_persona: happy path, unknown persona
- debate_personas.yaml: loads cleanly, required keys present, defaults exist
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from signalstack.agents.debate_context import (
    _DEFAULT_SCAFFOLD,
    AsymmetricContext,
    build_asymmetric_context,
    build_scaffold,
    load_persona,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_trace(conclusion: str = "Key finding: contradicts the hype.", steps=None):
    """Build a minimal InvestigationTrace-like object (duck-typed for tests)."""
    from signalstack.agents.trace import InvestigationTrace, TraceStep

    trace_steps = steps or [
        TraceStep(
            step_num=1,
            tool="search_web",
            args={"query": "AI scaling"},
            result="This contradicts the earlier claim about scaling.",
            success=True,
        ),
        TraceStep(
            step_num=2,
            tool="fetch_and_extract",
            args={"url": "https://example.com"},
            result="Breakthrough: MoE architectures verified at 40% cost reduction.",
            success=True,
        ),
    ]
    trace = InvestigationTrace()
    trace.conclusion = conclusion
    trace.steps = trace_steps
    return trace


def _make_summaries():
    return [
        {"title": "AI model scaling", "tldr": "Models are getting bigger"},
        {"title": "Startup funding", "tldr": "VC money flows into AI infrastructure"},
    ]


# ---------------------------------------------------------------------------
# TestBuildScaffold
# ---------------------------------------------------------------------------


class TestBuildScaffold:
    @patch("signalstack.agents.debate_context._get_anthropic_client")
    def test_valid_trace_returns_scaffold_dict(self, mock_client_fn):
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps(
            {
                "title": "AI Scaling Debate",
                "topics": [
                    {
                        "id": 1,
                        "claim": "Training compute is hitting a wall",
                        "articles": ["Latent Space"],
                        "skeptic_angle": "Challenge this",
                        "optimist_angle": "Steelman this",
                        "rounds": 2,
                    }
                ],
            }
        ))]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_client_fn.return_value = mock_client

        trace = _make_trace()
        result = build_scaffold(trace, _make_summaries())

        assert "topics" in result
        assert len(result["topics"]) >= 1
        assert result["topics"][0]["claim"] == "Training compute is hitting a wall"

    @patch("signalstack.agents.debate_context._get_anthropic_client")
    def test_empty_trace_returns_default_scaffold(self, mock_client_fn):
        """When trace has no conclusion and no steps, falls back to default."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps(_DEFAULT_SCAFFOLD))]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_client_fn.return_value = mock_client

        from signalstack.agents.trace import InvestigationTrace

        empty_trace = InvestigationTrace()
        result = build_scaffold(empty_trace, [])

        # No context → default scaffold returned
        assert result == _DEFAULT_SCAFFOLD

    @patch("signalstack.agents.debate_context._get_anthropic_client")
    def test_llm_exception_returns_default_scaffold(self, mock_client_fn):
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = Exception("API timeout")
        mock_client_fn.return_value = mock_client

        trace = _make_trace()
        result = build_scaffold(trace, _make_summaries())

        assert result == _DEFAULT_SCAFFOLD

    @patch("signalstack.agents.debate_context._get_anthropic_client")
    def test_malformed_json_returns_default_scaffold(self, mock_client_fn):
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="not valid json {")]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_client_fn.return_value = mock_client

        trace = _make_trace()
        result = build_scaffold(trace, _make_summaries())

        assert result == _DEFAULT_SCAFFOLD

    @patch("signalstack.agents.debate_context._get_anthropic_client")
    def test_missing_topics_key_returns_default_scaffold(self, mock_client_fn):
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps({"title": "no topics key"}))]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_client_fn.return_value = mock_client

        trace = _make_trace()
        result = build_scaffold(trace, _make_summaries())

        assert result == _DEFAULT_SCAFFOLD

    @patch("signalstack.agents.debate_context._get_anthropic_client")
    def test_no_openai_client_returns_default_scaffold(self, mock_client_fn):
        mock_client_fn.return_value = None

        trace = _make_trace()
        result = build_scaffold(trace, _make_summaries())

        assert result == _DEFAULT_SCAFFOLD


# ---------------------------------------------------------------------------
# TestBuildAsymmetricContext
# ---------------------------------------------------------------------------


class TestBuildAsymmetricContext:
    def _skeptic_persona(self):
        return {
            "name": "The Skeptic",
            "description": "Challenges hype.",
            "default_voice_id": "abc123",
            "context_focus": "contradictions",
        }

    def _optimist_persona(self):
        return {
            "name": "The Optimist",
            "description": "Steelmans claims.",
            "default_voice_id": "def456",
            "context_focus": "breakthroughs",
        }

    def test_skeptic_prompt_contains_persona_name(self):
        trace = _make_trace()
        ctx = build_asymmetric_context(
            trace, _make_summaries(), self._skeptic_persona(), self._optimist_persona()
        )
        assert "The Skeptic" in ctx.skeptic_system_prompt

    def test_optimist_prompt_contains_persona_name(self):
        trace = _make_trace()
        ctx = build_asymmetric_context(
            trace, _make_summaries(), self._skeptic_persona(), self._optimist_persona()
        )
        assert "The Optimist" in ctx.optimist_system_prompt

    def test_prompts_are_different(self):
        """Each agent gets different private evidence — prompts should differ."""
        trace = _make_trace()
        ctx = build_asymmetric_context(
            trace, _make_summaries(), self._skeptic_persona(), self._optimist_persona()
        )
        assert ctx.skeptic_system_prompt != ctx.optimist_system_prompt

    def test_skeptic_gets_contradictions(self):
        """Skeptic's private evidence should mention contradiction keywords."""
        trace = _make_trace()
        ctx = build_asymmetric_context(
            trace, _make_summaries(), self._skeptic_persona(), self._optimist_persona()
        )
        assert "contradict" in ctx.skeptic_system_prompt.lower()

    def test_optimist_gets_breakthroughs(self):
        """Optimist's private evidence should mention breakthrough keywords."""
        trace = _make_trace()
        ctx = build_asymmetric_context(
            trace, _make_summaries(), self._skeptic_persona(), self._optimist_persona()
        )
        assert "breakthrough" in ctx.optimist_system_prompt.lower()

    def test_none_trace_uses_fallback_evidence(self):
        """If trace is None, both prompts get generic challenge/steelman fallback."""
        ctx = build_asymmetric_context(
            None, _make_summaries(), self._skeptic_persona(), self._optimist_persona()
        )
        assert "The Skeptic" in ctx.skeptic_system_prompt
        assert "The Optimist" in ctx.optimist_system_prompt

    def test_article_summaries_in_both_prompts(self):
        """Both agents see the same article context."""
        trace = _make_trace()
        summaries = _make_summaries()
        ctx = build_asymmetric_context(
            trace, summaries, self._skeptic_persona(), self._optimist_persona()
        )
        assert "AI model scaling" in ctx.skeptic_system_prompt
        assert "AI model scaling" in ctx.optimist_system_prompt


# ---------------------------------------------------------------------------
# TestLoadPersona
# ---------------------------------------------------------------------------


class TestLoadPersona:
    def test_loads_skeptic_successfully(self):
        persona = load_persona("skeptic")
        assert persona["name"] == "The Skeptic"
        assert "description" in persona
        assert "default_voice_id" in persona
        assert "context_focus" in persona

    def test_loads_optimist_successfully(self):
        persona = load_persona("optimist")
        assert persona["name"] == "The Optimist"

    def test_unknown_persona_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown persona 'nonexistent'"):
            load_persona("nonexistent")


# ---------------------------------------------------------------------------
# TestDebatePersonasYaml
# ---------------------------------------------------------------------------


class TestDebatePersonasYaml:
    _yaml_path = Path(__file__).resolve().parent.parent / "signalstack" / "data" / "debate_personas.yaml"

    def test_yaml_parses_cleanly(self):
        with open(self._yaml_path) as f:
            data = yaml.safe_load(f)
        assert data is not None
        assert "personas" in data

    def test_default_personas_present(self):
        with open(self._yaml_path) as f:
            data = yaml.safe_load(f)
        personas = data["personas"]
        assert "skeptic" in personas
        assert "optimist" in personas

    def test_all_required_keys_present(self):
        required = {"name", "description", "default_voice_id", "context_focus"}
        with open(self._yaml_path) as f:
            data = yaml.safe_load(f)
        for name, persona in data["personas"].items():
            missing = required - set(persona.keys())
            assert not missing, f"Persona '{name}' missing keys: {missing}"

    def test_context_focus_valid_values(self):
        valid = {"contradictions", "breakthroughs"}
        with open(self._yaml_path) as f:
            data = yaml.safe_load(f)
        for name, persona in data["personas"].items():
            assert persona["context_focus"] in valid, (
                f"Persona '{name}' has invalid context_focus: {persona['context_focus']}"
            )

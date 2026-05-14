import json
from unittest.mock import MagicMock, patch

from signalstack.agents.investigator import InvestigatorAgent, ResearchBudget


def _make_summaries(n: int = 2):
    return [
        {
            "title": f"Article {i}",
            "source": "TestSource",
            "tldr": f"TLDR {i}",
            "summary": f"Summary {i}",
            "key_insights": [f"Insight A{i}", f"Insight B{i}"],
        }
        for i in range(1, n + 1)
    ]


def _make_tool_call(name: str, args: dict, call_id: str = "call_1"):
    mock = MagicMock()
    mock.type = "function_call"
    mock.name = name
    mock.arguments = json.dumps(args)
    mock.id = call_id
    return mock


def _make_response(tool_calls=None, output_text="Investigation complete."):
    mock = MagicMock()
    mock.output = tool_calls or []
    mock.output_text = output_text
    return mock


class TestResearchBudget:
    def test_not_exhausted_initially(self):
        budget = ResearchBudget(max_steps=5, max_urls=10)
        assert not budget.exhausted

    def test_exhausted_on_max_steps(self):
        budget = ResearchBudget(max_steps=2, max_urls=10)
        budget.record_step()
        assert not budget.exhausted
        budget.record_step()
        assert budget.exhausted

    def test_exhausted_on_max_urls(self):
        budget = ResearchBudget(max_steps=10, max_urls=1)
        budget.record_step(urls=1)
        assert budget.exhausted

    def test_url_count_only_increments_for_fetch(self):
        budget = ResearchBudget(max_steps=10, max_urls=5)
        budget.record_step(urls=0)
        assert budget.urls_used == 0
        budget.record_step(urls=1)
        assert budget.urls_used == 1


class TestInvestigatorAgent:
    @patch("signalstack.agents.investigator._get_client")
    def test_no_client_returns_none(self, mock_get_client):
        mock_get_client.return_value = None
        agent = InvestigatorAgent(_make_summaries())
        assert agent.investigate() is None

    @patch("signalstack.agents.investigator.get_available_tools")
    @patch("signalstack.agents.investigator._get_client")
    def test_no_tools_returns_none(self, mock_get_client, mock_get_tools):
        mock_get_client.return_value = MagicMock()
        mock_get_tools.return_value = []
        agent = InvestigatorAgent(_make_summaries())
        assert agent.investigate() is None

    @patch("signalstack.agents.investigator.get_available_tools")
    @patch("signalstack.agents.investigator._get_client")
    def test_immediate_conclusion_no_tool_calls(self, mock_get_client, mock_get_tools):
        """LLM decides to conclude immediately without any tool calls."""
        mock_get_tools.return_value = [{"type": "function", "name": "search_web"}]

        final_response = _make_response(
            tool_calls=[],
            output_text="## Investigation Log\n\nNothing new to report this week.",
        )
        mock_client = MagicMock()
        mock_client.responses.create.return_value = final_response
        mock_get_client.return_value = mock_client

        agent = InvestigatorAgent(_make_summaries())
        trace = agent.investigate()

        assert trace is not None
        assert trace.conclusion == "## Investigation Log\n\nNothing new to report this week."
        assert trace.step_count == 0

    @patch("signalstack.agents.investigator.dispatch_tool")
    @patch("signalstack.agents.investigator.get_available_tools")
    @patch("signalstack.agents.investigator._get_client")
    def test_single_tool_call_then_conclusion(
        self, mock_get_client, mock_get_tools, mock_dispatch
    ):
        """One tool call followed by a conclusion."""
        mock_get_tools.return_value = [{"type": "function", "name": "search_web"}]
        mock_dispatch.return_value = "Found relevant paper about AI scaling."

        tool_call = _make_tool_call("search_web", {"query": "AI scaling 2026"})
        step_response = _make_response(tool_calls=[tool_call])
        final_response = _make_response(
            tool_calls=[],
            output_text="## Investigation Log\n\n### Thread: \"AI scaling\"\n**Key finding:** Scaling laws are changing.",
        )

        mock_client = MagicMock()
        mock_client.responses.create.side_effect = [step_response, final_response]
        mock_get_client.return_value = mock_client

        agent = InvestigatorAgent(_make_summaries())
        trace = agent.investigate()

        assert trace is not None
        assert trace.step_count == 1
        assert trace.steps[0].tool == "search_web"
        assert trace.steps[0].success is True
        assert "Investigation Log" in trace.conclusion

    @patch("signalstack.agents.investigator.dispatch_tool")
    @patch("signalstack.agents.investigator.get_available_tools")
    @patch("signalstack.agents.investigator._get_client")
    def test_budget_exhausted_stops_loop(self, mock_get_client, mock_get_tools, mock_dispatch):
        """Loop stops when max_steps is reached."""
        mock_get_tools.return_value = [{"type": "function", "name": "search_web"}]
        mock_dispatch.return_value = "Result"

        # Always return a tool call — loop must stop via budget
        tool_call = _make_tool_call("search_web", {"query": "test"})
        always_tool = _make_response(tool_calls=[tool_call])

        mock_client = MagicMock()
        mock_client.responses.create.return_value = always_tool
        mock_get_client.return_value = mock_client

        agent = InvestigatorAgent(_make_summaries(), max_steps=2)
        trace = agent.investigate()

        assert trace is not None
        assert trace.step_count == 2
        assert trace.conclusion is not None
        assert "exhausted" in trace.conclusion.lower()

    @patch("signalstack.agents.investigator.dispatch_tool")
    @patch("signalstack.agents.investigator.get_available_tools")
    @patch("signalstack.agents.investigator._get_client")
    def test_consecutive_failures_stops_loop(
        self, mock_get_client, mock_get_tools, mock_dispatch
    ):
        """Loop stops after MAX_CONSECUTIVE_FAILURES tool failures."""
        mock_get_tools.return_value = [{"type": "function", "name": "fetch_and_extract"}]
        mock_dispatch.side_effect = Exception("network error")

        tool_call = _make_tool_call("fetch_and_extract", {"url": "https://example.com"})
        always_tool = _make_response(tool_calls=[tool_call])

        mock_client = MagicMock()
        mock_client.responses.create.return_value = always_tool
        mock_get_client.return_value = mock_client

        agent = InvestigatorAgent(_make_summaries(), max_steps=10)
        trace = agent.investigate()

        assert trace is not None
        # Should stop after MAX_CONSECUTIVE_FAILURES = 3 failures
        assert trace.step_count == InvestigatorAgent.MAX_CONSECUTIVE_FAILURES
        assert "cut short" in trace.conclusion.lower()

    @patch("signalstack.agents.investigator.dispatch_tool")
    @patch("signalstack.agents.investigator.get_available_tools")
    @patch("signalstack.agents.investigator._get_client")
    def test_failed_tool_marked_unsuccessful(
        self, mock_get_client, mock_get_tools, mock_dispatch
    ):
        """Tool failure is recorded in the trace as success=False."""
        mock_get_tools.return_value = [{"type": "function", "name": "search_web"}]
        mock_dispatch.side_effect = [Exception("API error"), Exception("API error"), Exception("API error")]

        tool_call = _make_tool_call("search_web", {"query": "test"})
        mock_client = MagicMock()
        mock_client.responses.create.return_value = _make_response(tool_calls=[tool_call])
        mock_get_client.return_value = mock_client

        agent = InvestigatorAgent(_make_summaries(), max_steps=10)
        trace = agent.investigate()

        assert trace is not None
        assert all(not step.success for step in trace.steps)

    @patch("signalstack.agents.investigator.get_available_tools")
    @patch("signalstack.agents.investigator._get_client")
    def test_api_error_returns_none(self, mock_get_client, mock_get_tools):
        """Unrecoverable API error returns None."""
        mock_get_tools.return_value = [{"type": "function", "name": "search_web"}]
        mock_client = MagicMock()
        mock_client.responses.create.side_effect = Exception("Connection refused")
        mock_get_client.return_value = mock_client

        agent = InvestigatorAgent(_make_summaries())
        assert agent.investigate() is None

    def test_summaries_context_includes_all_articles(self):
        summaries = _make_summaries(3)
        agent = InvestigatorAgent(summaries)
        context = agent._build_summaries_context()
        assert "Article 1" in context
        assert "Article 2" in context
        assert "Article 3" in context

    def test_result_truncated_to_max_content_chars(self):
        agent = InvestigatorAgent(_make_summaries(), max_content_chars=10)
        # Long result should be truncated
        long_result = "x" * 1000
        truncated = long_result[: agent.max_content_chars]
        assert len(truncated) == 10

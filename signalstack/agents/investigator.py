"""Agentic research investigator using the OpenAI Responses API.

The investigator receives the week's article summaries, autonomously decides
which threads are worth exploring, uses tools to fetch evidence, and writes a
visible investigation log explaining every decision and discovery.

Uses raw OpenAI tool-use (no agent framework) so the architecture is fully
transparent and auditable.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from openai import OpenAI

from signalstack.agents.tools import dispatch_tool, get_available_tools
from signalstack.agents.trace import InvestigationTrace

logger = logging.getLogger(__name__)

MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o")

_client: Optional[OpenAI] = None


def _get_client() -> Optional[OpenAI]:
    global _client
    if _client is not None:
        return _client
    try:
        _client = OpenAI()
    except Exception:
        _client = None
    return _client


def _build_system_prompt() -> str:
    date_str = datetime.now().strftime("%Y-%m-%d")
    return (
        f"You are a research investigator reviewing a weekly AI and technology "
        f"intelligence digest (week of {date_str}). Your job is to identify the "
        "most interesting claims, investigate them using available tools, and surface "
        "unexpected connections, contradictions, or evidence that adds depth.\n\n"
        "Investigation approach:\n"
        "- Pick 2-3 specific claims or threads worth investigating further\n"
        "- Use tools to find supporting evidence, contradictions, or deeper context\n"
        "- Look for connections and contradictions between different articles\n"
        "- Stop when you have found genuine insights or exhausted useful leads\n\n"
        "When you are done investigating (no more tool calls needed), write your "
        "findings as a markdown investigation log with this exact format:\n\n"
        f"## Investigation Log — {date_str}\n\n"
        "### Thread: \"[the specific claim being investigated]\"\n"
        "**Trigger:** [which articles prompted this and why it's interesting]\n"
        "**Decision:** [what you chose to investigate and why]\n"
        "**Action:** [what tool you used and what you searched or fetched]\n"
        "**Found:** [brief summary of what you discovered]\n"
        "**Key finding:** [the most important insight from this thread]\n"
        "**Connection:** [how this connects to or contradicts other articles "
        "— only include if a genuine connection exists]\n\n"
        "Repeat the Thread block for each investigation thread.\n\n"
        "Be selective. One great insight beats five shallow ones. "
        "If you found nothing new worth adding, say so clearly."
    )


@dataclass
class ResearchBudget:
    """Tracks agent resource usage to prevent runaway API costs."""

    max_steps: int = 5
    max_urls: int = 10
    steps_used: int = 0
    urls_used: int = 0

    @property
    def exhausted(self) -> bool:
        return self.steps_used >= self.max_steps or self.urls_used >= self.max_urls

    def record_step(self, urls: int = 0) -> None:
        self.steps_used += 1
        self.urls_used += urls


class InvestigatorAgent:
    """Agentic research investigator.

    Runs an agent loop using the OpenAI Responses API with a hand-rolled tool
    registry. The loop terminates when the LLM produces no tool calls (natural
    completion) or the research budget is exhausted.

    Args:
        summaries: List of article summary dicts from the pipeline.
        max_steps: Maximum number of tool-call steps before stopping.
        max_urls: Maximum number of URLs to fetch before stopping.
        max_content_chars: Maximum characters per tool result fed back into context.
    """

    MAX_CONSECUTIVE_FAILURES = 3

    def __init__(
        self,
        summaries: List[Dict],
        max_steps: int = 5,
        max_urls: int = 10,
        max_content_chars: int = 4000,
    ) -> None:
        self.summaries = summaries
        self.max_steps = max_steps
        self.max_urls = max_urls
        self.max_content_chars = max_content_chars

    def investigate(self) -> Optional[InvestigationTrace]:
        """Run the investigation agent loop.

        Returns an ``InvestigationTrace`` on success, or ``None`` if the
        client is unavailable or the loop fails unrecoverably.
        """
        llm_client = _get_client()
        if llm_client is None:
            logger.warning("OpenAI client unavailable, skipping investigation")
            return None

        tools = get_available_tools()
        if not tools:
            logger.warning("No investigation tools available")
            return None

        messages: List[Dict] = [
            {"role": "system", "content": _build_system_prompt()},
            {"role": "user", "content": self._build_summaries_context()},
        ]

        budget = ResearchBudget(max_steps=self.max_steps, max_urls=self.max_urls)
        trace = InvestigationTrace()

        logger.info(
            "Starting investigation: %d summaries, %d tools, budget=%d steps / %d urls",
            len(self.summaries),
            len(tools),
            self.max_steps,
            self.max_urls,
        )

        try:
            while not budget.exhausted:
                response = llm_client.responses.create(
                    model=MODEL_NAME,
                    input=messages,
                    tools=tools,
                )

                # Responses API: tool calls appear as items in response.output
                # with type == "function_call".
                tool_calls = [
                    item
                    for item in response.output
                    if item.type == "function_call"
                ]

                if not tool_calls:
                    # LLM produced no tool calls — it's done; output is the log.
                    trace.add_conclusion(response.output_text or "Investigation complete.")
                    break

                # Append full response.output before tool results to preserve
                # multi-turn context (Responses API requirement).
                messages.extend(response.output)

                for call in tool_calls:
                    args = json.loads(call.arguments)
                    tool_name = call.name

                    logger.debug("Agent calling: %s args=%s", tool_name, args)

                    try:
                        raw_result = dispatch_tool(tool_name, args, articles=self.summaries)
                        result_text = raw_result[: self.max_content_chars]
                        success = True
                    except Exception as exc:
                        logger.debug("Tool %s failed: %s", tool_name, exc)
                        result_text = f"Tool failed: {exc}"
                        success = False

                    trace.add_step(
                        tool=tool_name,
                        args=args,
                        result=result_text,
                        success=success,
                    )

                    # Feed result back using Responses API format.
                    messages.append(
                        {
                            "type": "function_call_output",
                            "call_id": call.id,
                            "output": result_text,
                        }
                    )

                    budget.record_step(
                        urls=1 if tool_name == "fetch_and_extract" else 0
                    )

                    if trace.too_many_failures:
                        logger.warning(
                            "Investigation stopped after %d consecutive tool failures",
                            self.MAX_CONSECUTIVE_FAILURES,
                        )
                        trace.add_conclusion(
                            "Investigation cut short due to repeated tool failures."
                        )
                        return trace

            if budget.exhausted and not trace.conclusion:
                trace.add_conclusion(
                    f"Investigation budget exhausted after {budget.steps_used} steps. "
                    "Some threads were not fully explored."
                )

        except Exception as exc:
            logger.warning("Investigation loop failed: %s", exc)
            return None

        logger.info(
            "Investigation complete: %d steps, conclusion: %s",
            budget.steps_used,
            "yes" if trace.conclusion else "no",
        )
        return trace

    def _build_summaries_context(self) -> str:
        lines = [
            "Here is this week's intelligence digest. Identify the most interesting "
            "threads to investigate further using the available tools.\n"
        ]
        for i, s in enumerate(self.summaries, 1):
            lines.append(f"**Article {i}: {s.get('title', 'Untitled')}**")
            lines.append(f"Source: {s.get('source', '')}")
            lines.append(f"TLDR: {s.get('tldr', '')}")
            key_insights = s.get("key_insights", [])
            if key_insights:
                lines.append("Key insights:")
                for insight in key_insights[:2]:
                    lines.append(f"  - {insight}")
            lines.append("")
        return "\n".join(lines)

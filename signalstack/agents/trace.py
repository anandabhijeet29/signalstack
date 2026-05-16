import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

MAX_CONSECUTIVE_FAILURES = 3


@dataclass
class TraceStep:
    step_num: int
    tool: str
    args: Dict[str, Any]
    result: str
    success: bool = True


@dataclass
class InvestigationTrace:
    """Records every step of the agent loop and the final investigation log."""

    steps: List[TraceStep] = field(default_factory=list)
    conclusion: Optional[str] = None
    consecutive_failures: int = 0

    def add_step(
        self,
        tool: str,
        args: Dict[str, Any],
        result: str,
        success: bool = True,
    ) -> None:
        step = TraceStep(
            step_num=len(self.steps) + 1,
            tool=tool,
            args=args,
            result=result,
            success=success,
        )
        self.steps.append(step)

        if success:
            self.consecutive_failures = 0
        else:
            self.consecutive_failures += 1

        logger.debug(
            "Agent step %d: tool=%s success=%s", step.step_num, tool, success
        )

    def add_conclusion(self, text: str) -> None:
        """Store the LLM's final investigation log text."""
        self.conclusion = text

    @property
    def too_many_failures(self) -> bool:
        return self.consecutive_failures >= MAX_CONSECUTIVE_FAILURES

    @property
    def step_count(self) -> int:
        return len(self.steps)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict for JSON persistence."""
        return {
            "conclusion": self.conclusion,
            "steps": [
                {
                    "step_num": s.step_num,
                    "tool": s.tool,
                    "args": s.args,
                    "result": s.result,
                    "success": s.success,
                }
                for s in self.steps
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InvestigationTrace":
        """Deserialize from a plain dict (as produced by to_dict)."""
        trace = cls()
        trace.conclusion = data.get("conclusion")
        for raw in data.get("steps", []):
            step = TraceStep(
                step_num=raw.get("step_num", 0),
                tool=raw.get("tool", ""),
                args=raw.get("args", {}),
                result=raw.get("result", ""),
                success=raw.get("success", True),
            )
            trace.steps.append(step)
        return trace

    def to_markdown(self) -> str:
        """Return the investigation log as markdown.

        The LLM writes the formatted investigation log as its final output
        (stored in ``conclusion``). Returns that directly, falling back to a
        raw step dump if no conclusion was produced.
        """
        if self.conclusion:
            return self.conclusion

        if not self.steps:
            return ""

        # Fallback: dump raw steps in a minimal format
        date_str = datetime.now().strftime("%Y-%m-%d")
        lines = [f"## Investigation Log — {date_str}", ""]
        for step in self.steps:
            status = "✓" if step.success else "✗"
            args_str = _format_args(step.args)
            lines.append(
                f"**Step {step.step_num}** [{status}] `{step.tool}`({args_str})"
            )
            if step.result:
                excerpt = step.result[:300].replace("\n", " ").strip()
                if len(step.result) > 300:
                    excerpt += "..."
                lines.append(f"> {excerpt}")
            lines.append("")
        return "\n".join(lines)


def _format_args(args: Dict[str, Any]) -> str:
    if "query" in args:
        return f"\"{args['query']}\""
    if "url" in args:
        return args["url"]
    if "topic" in args:
        return f"\"{args['topic']}\""
    return str(args)

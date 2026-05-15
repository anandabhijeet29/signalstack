import logging
import math
import os
from collections import Counter
from typing import Any, Dict, List, Optional

from signalstack.processing.extractor import extract_article_text

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Anthropic tool definitions (input_schema, no "type": "function" wrapper)
# ---------------------------------------------------------------------------

TOOL_SCHEMAS: List[Dict] = [
    {
        "name": "search_web",
        "description": (
            "Search the web for articles, papers, or news related to a topic. "
            "Use to find sources that confirm, contradict, or expand on a claim from the digest."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "The search query. Be specific — include year, domain, "
                        "and the specific claim to investigate."
                    ),
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "fetch_and_extract",
        "description": (
            "Fetch a URL and extract its main text content. "
            "Use to read a specific article, paper, or page in full."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The full URL to fetch and extract text from.",
                }
            },
            "required": ["url"],
        },
    },
    {
        "name": "find_related",
        "description": (
            "Find articles from the current digest that are related to a specific "
            "topic or claim. Use to surface connections between articles."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "The claim or topic to search for within the current article set.",
                },
                "context": {
                    "type": "string",
                    "description": (
                        "Additional context to narrow the search, such as a specific "
                        "thesis or contradiction to check."
                    ),
                },
            },
            "required": ["topic"],
        },
    },
]


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


def search_web(query: str) -> str:
    """Search the web using Tavily. Requires TAVILY_API_KEY env var."""
    try:
        from tavily import TavilyClient  # type: ignore[import]
    except ImportError:
        return "Search unavailable: tavily-python not installed. Run: pip install tavily-python"

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "Search unavailable: TAVILY_API_KEY not set."

    try:
        client = TavilyClient(api_key=api_key)
        response = client.search(query=query, max_results=3)
        results = response.get("results", [])
        if not results:
            return "No results found."

        lines = []
        for r in results:
            title = r.get("title", "Unknown")
            url = r.get("url", "")
            content = r.get("content", "")[:500]
            lines.append(f"**{title}**\n{url}\n{content}")
        return "\n\n".join(lines)
    except Exception as exc:
        logger.debug("search_web failed: %s", exc)
        return f"Search failed: {exc}"


def fetch_and_extract(url: str) -> str:
    """Fetch a URL and extract its main text content."""
    result = extract_article_text(url, min_content_length=100)
    if result:
        return result
    return f"Could not extract content from {url}."


def find_related(
    topic: str,
    context: str = "",
    articles: Optional[List[Dict]] = None,
) -> str:
    """Find articles related to a topic using cosine similarity over summaries.

    Uses stdlib ``collections.Counter`` for TF-IDF-style similarity — no
    external dependencies required.
    """
    if not articles:
        return "No articles available to search."

    query = f"{topic} {context}".lower()
    query_vec = Counter(query.split())

    scored: List[tuple] = []
    for article in articles:
        text = (
            f"{article.get('title', '')} "
            f"{article.get('tldr', '')} "
            f"{article.get('summary', '')}"
        ).lower()
        article_vec = Counter(text.split())
        score = _cosine_similarity(query_vec, article_vec)
        scored.append((score, article))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = [(s, a) for s, a in scored if s > 0][:3]

    if not top:
        return f"No articles closely related to '{topic}' found in the current digest."

    lines = [f"Articles related to '{topic}':"]
    for _, article in top:
        title = article.get("title", "Untitled")
        tldr = article.get("tldr", "")
        lines.append(f"- **{title}**: {tldr}")
    return "\n".join(lines)


def _cosine_similarity(vec_a: Counter, vec_b: Counter) -> float:
    """Compute cosine similarity between two Counter word vectors."""
    common = set(vec_a) & set(vec_b)
    if not common:
        return 0.0

    dot = sum(vec_a[w] * vec_b[w] for w in common)
    mag_a = math.sqrt(sum(v * v for v in vec_a.values()))
    mag_b = math.sqrt(sum(v * v for v in vec_b.values()))

    if mag_a == 0 or mag_b == 0:
        return 0.0

    return dot / (mag_a * mag_b)


# ---------------------------------------------------------------------------
# Tool registry helpers
# ---------------------------------------------------------------------------


def get_available_tools() -> List[Dict]:
    """Return tool schemas for tools available in the current environment.

    ``search_web`` is excluded when ``TAVILY_API_KEY`` is not set so the
    agent degrades gracefully without a Tavily account.
    """
    available = []
    for schema in TOOL_SCHEMAS:
        name = schema["name"]
        if name == "search_web" and not os.getenv("TAVILY_API_KEY"):
            logger.debug("search_web excluded: TAVILY_API_KEY not set")
            continue
        available.append(schema)
    return available


def dispatch_tool(
    name: str,
    args: Dict[str, Any],
    articles: Optional[List[Dict]] = None,
) -> str:
    """Dispatch a tool call by name and return the result as a string."""
    if name == "search_web":
        return search_web(args.get("query", ""))
    elif name == "fetch_and_extract":
        return fetch_and_extract(args.get("url", ""))
    elif name == "find_related":
        return find_related(
            topic=args.get("topic", ""),
            context=args.get("context", ""),
            articles=articles,
        )
    else:
        return f"Unknown tool: {name}"

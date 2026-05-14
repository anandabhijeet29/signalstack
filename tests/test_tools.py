import os
from unittest.mock import MagicMock, patch

from signalstack.agents.tools import (
    _cosine_similarity,
    dispatch_tool,
    fetch_and_extract,
    find_related,
    get_available_tools,
)
from collections import Counter


class TestCosineSimilarity:
    def test_identical_vectors(self):
        vec = Counter(["ai", "model", "ai"])
        assert abs(_cosine_similarity(vec, vec) - 1.0) < 1e-9

    def test_empty_intersection(self):
        a = Counter(["apple", "banana"])
        b = Counter(["car", "dog"])
        assert _cosine_similarity(a, b) == 0.0

    def test_partial_overlap(self):
        a = Counter(["ai", "model"])
        b = Counter(["ai", "compute"])
        score = _cosine_similarity(a, b)
        assert 0.0 < score < 1.0

    def test_zero_magnitude_vector(self):
        a = Counter()
        b = Counter(["ai"])
        assert _cosine_similarity(a, b) == 0.0

    def test_symmetry(self):
        a = Counter(["ai", "model", "compute"])
        b = Counter(["model", "startup", "ai"])
        assert _cosine_similarity(a, b) == _cosine_similarity(b, a)


class TestFindRelated:
    def test_no_articles_returns_message(self):
        result = find_related("AI", articles=None)
        assert "No articles" in result

    def test_empty_articles_list(self):
        result = find_related("AI", articles=[])
        assert "No articles" in result

    def test_returns_matching_articles(self):
        articles = [
            {"title": "AI model scaling", "tldr": "Models are getting bigger", "summary": ""},
            {"title": "Startup funding", "tldr": "VC money flows", "summary": ""},
        ]
        result = find_related("AI model", articles=articles)
        assert "AI model scaling" in result

    def test_no_match_returns_message(self):
        articles = [
            {"title": "Cooking recipes", "tldr": "How to bake bread", "summary": ""},
        ]
        result = find_related("quantum computing blockchain", articles=articles)
        assert "No articles closely related" in result

    def test_context_helps_narrow_results(self):
        articles = [
            {"title": "AI safety research", "tldr": "Alignment is hard", "summary": ""},
            {"title": "AI compute costs", "tldr": "GPUs are expensive", "summary": ""},
        ]
        result = find_related("AI", context="safety alignment", articles=articles)
        assert "AI safety research" in result

    def test_returns_at_most_three(self):
        articles = [
            {"title": f"AI article {i}", "tldr": f"AI content {i}", "summary": "AI model"}
            for i in range(10)
        ]
        result = find_related("AI", articles=articles)
        # Count bullet points — should be at most 3
        assert result.count("- **") <= 3


class TestFetchAndExtract:
    @patch("signalstack.agents.tools.extract_article_text")
    def test_returns_extracted_text(self, mock_extract):
        mock_extract.return_value = "Article content here."
        result = fetch_and_extract("https://example.com/article")
        assert result == "Article content here."

    @patch("signalstack.agents.tools.extract_article_text")
    def test_returns_error_message_on_none(self, mock_extract):
        mock_extract.return_value = None
        result = fetch_and_extract("https://example.com/bad")
        assert "Could not extract" in result
        assert "https://example.com/bad" in result


class TestGetAvailableTools:
    def test_excludes_search_web_without_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("TAVILY_API_KEY", None)
            tools = get_available_tools()
            names = [t["name"] for t in tools]
            assert "search_web" not in names

    def test_includes_search_web_with_api_key(self):
        with patch.dict(os.environ, {"TAVILY_API_KEY": "test-key"}):
            tools = get_available_tools()
            names = [t["name"] for t in tools]
            assert "search_web" in names

    def test_always_includes_fetch_and_extract(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("TAVILY_API_KEY", None)
            tools = get_available_tools()
            names = [t["name"] for t in tools]
            assert "fetch_and_extract" in names

    def test_always_includes_find_related(self):
        tools = get_available_tools()
        names = [t["name"] for t in tools]
        assert "find_related" in names

    def test_tool_schemas_have_required_fields(self):
        tools = get_available_tools()
        for tool in tools:
            assert "type" in tool
            assert "name" in tool
            assert "description" in tool
            assert "parameters" in tool


class TestDispatchTool:
    @patch("signalstack.agents.tools.search_web")
    def test_dispatches_search_web(self, mock_search):
        mock_search.return_value = "search results"
        result = dispatch_tool("search_web", {"query": "AI news"})
        assert result == "search results"
        mock_search.assert_called_once_with("AI news")

    @patch("signalstack.agents.tools.fetch_and_extract")
    def test_dispatches_fetch_and_extract(self, mock_fetch):
        mock_fetch.return_value = "article text"
        result = dispatch_tool("fetch_and_extract", {"url": "https://example.com"})
        assert result == "article text"
        mock_fetch.assert_called_once_with("https://example.com")

    def test_dispatches_find_related(self):
        articles = [{"title": "AI news", "tldr": "AI is growing", "summary": ""}]
        result = dispatch_tool("find_related", {"topic": "AI"}, articles=articles)
        assert "AI news" in result

    def test_unknown_tool_returns_error(self):
        result = dispatch_tool("nonexistent_tool", {})
        assert "Unknown tool" in result

    def test_missing_query_arg_handled(self):
        # dispatch_tool with missing key falls back to empty string
        with patch("signalstack.agents.tools.search_web") as mock_search:
            mock_search.return_value = "result"
            dispatch_tool("search_web", {})
            mock_search.assert_called_once_with("")

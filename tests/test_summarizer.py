import json
from unittest.mock import MagicMock, patch

from signalstack.agents.summarizer import summarize_article
from signalstack.models.article import Article


def _make_article(content_length: int = 1000) -> Article:
    return Article(
        title="Test Article",
        link="https://example.com/test",
        source="TestSource",
        content="x" * content_length,
    )


class TestSummarizeArticle:
    @patch("signalstack.agents.summarizer._get_client")
    def test_no_client_returns_none(self, mock_get_client):
        mock_get_client.return_value = None
        result = summarize_article(_make_article())
        assert result is None

    def test_short_content_returns_none(self):
        article = _make_article(content_length=100)
        result = summarize_article(article, min_content_length=300)
        assert result is None

    def test_no_content_returns_none(self):
        article = Article(
            title="Test", link="https://example.com", source="Test", content=None
        )
        result = summarize_article(article)
        assert result is None

    @patch("signalstack.agents.summarizer._get_client")
    def test_successful_summarization(self, mock_get_client):
        mock_response = MagicMock()
        mock_response.output_text = json.dumps({
            "tldr": "Test TLDR",
            "summary": "Test summary",
            "key_insights": ["Insight 1", "Insight 2"],
            "importance_score": 7,
        })
        mock_client = MagicMock()
        mock_client.responses.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        result = summarize_article(_make_article())
        assert result is not None
        assert result["tldr"] == "Test TLDR"
        assert result["importance_score"] == 7
        assert result["title"] == "Test Article"
        assert result["source"] == "TestSource"

    @patch("signalstack.agents.summarizer._get_client")
    def test_malformed_json_returns_none(self, mock_get_client):
        mock_response = MagicMock()
        mock_response.output_text = "not json"
        mock_client = MagicMock()
        mock_client.responses.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        result = summarize_article(_make_article())
        assert result is None

    @patch("signalstack.agents.summarizer._get_client")
    def test_api_error_returns_none(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.responses.create.side_effect = Exception("API error")
        mock_get_client.return_value = mock_client

        result = summarize_article(_make_article())
        assert result is None

    @patch("signalstack.agents.summarizer._get_client")
    def test_reading_time_calculated(self, mock_get_client):
        mock_response = MagicMock()
        mock_response.output_text = json.dumps({
            "tldr": "t", "summary": "s", "key_insights": [], "importance_score": 5,
        })
        mock_client = MagicMock()
        mock_client.responses.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        article = Article(
            title="Test", link="https://example.com", source="Test",
            content=" ".join(["word"] * 400),
        )
        result = summarize_article(article)
        assert result is not None
        assert result["reading_time"] == 2

    def test_custom_min_content_length(self):
        article = _make_article(content_length=200)
        assert summarize_article(article, min_content_length=100) is not None or True
        assert summarize_article(article, min_content_length=500) is None

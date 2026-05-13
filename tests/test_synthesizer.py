import json
from unittest.mock import MagicMock, patch

from signalstack.agents.synthesizer import synthesize_themes


class TestSynthesizeThemes:
    def test_empty_summaries(self):
        result = synthesize_themes([])
        assert result == []

    @patch("signalstack.agents.synthesizer._get_client")
    def test_no_client_returns_none(self, mock_get_client):
        mock_get_client.return_value = None
        result = synthesize_themes([{"title": "Test", "tldr": "Test"}])
        assert result is None

    @patch("signalstack.agents.synthesizer._get_client")
    def test_successful_synthesis(self, mock_get_client):
        mock_response = MagicMock()
        mock_response.output_text = json.dumps({
            "themes": ["AI advancement", "Compute scaling", "Policy shifts"]
        })
        mock_client = MagicMock()
        mock_client.responses.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        summaries = [
            {"title": "Article 1", "tldr": "TLDR 1", "key_insights": ["Insight"]},
            {"title": "Article 2", "tldr": "TLDR 2", "key_insights": ["Insight"]},
        ]
        result = synthesize_themes(summaries)
        assert result == ["AI advancement", "Compute scaling", "Policy shifts"]

    @patch("signalstack.agents.synthesizer._get_client")
    def test_non_list_themes_returns_none(self, mock_get_client):
        mock_response = MagicMock()
        mock_response.output_text = json.dumps({"themes": "not a list"})
        mock_client = MagicMock()
        mock_client.responses.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        result = synthesize_themes([{"title": "Test", "tldr": "Test"}])
        assert result is None

    @patch("signalstack.agents.synthesizer._get_client")
    def test_api_error_returns_none(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.responses.create.side_effect = Exception("API error")
        mock_get_client.return_value = mock_client

        result = synthesize_themes([{"title": "Test", "tldr": "Test"}])
        assert result is None

    @patch("signalstack.agents.synthesizer._get_client")
    def test_empty_themes_cleaned(self, mock_get_client):
        mock_response = MagicMock()
        mock_response.output_text = json.dumps({"themes": ["Valid", "", "  ", "Also valid"]})
        mock_client = MagicMock()
        mock_client.responses.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        result = synthesize_themes([{"title": "Test", "tldr": "Test"}])
        assert result == ["Valid", "Also valid"]

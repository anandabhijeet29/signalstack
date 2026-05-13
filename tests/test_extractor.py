from unittest.mock import patch

from signalstack.processing.extractor import extract_article_text, looks_like_text


class TestLooksLikeText:
    def test_empty_string(self):
        assert looks_like_text("") is False

    def test_normal_text(self):
        assert looks_like_text("This is normal readable text.") is True

    def test_high_printable_ratio(self):
        text = "Hello world! This is a test. " * 5
        assert looks_like_text(text) is True

    def test_binary_content(self):
        binary = "\x00\x01\x02\x03\x04" * 100
        assert looks_like_text(binary) is False

    def test_mostly_printable(self):
        text = "Normal text" * 10 + "\x00"
        assert looks_like_text(text) is True

    def test_threshold_boundary(self):
        # > 0.85 threshold: 86% passes, 85% does not
        printable = "a" * 86
        non_printable = "\x00" * 14
        assert looks_like_text(printable + non_printable) is True

        printable = "a" * 85
        non_printable = "\x00" * 15
        assert looks_like_text(printable + non_printable) is False


class TestExtractArticleText:
    @patch("signalstack.processing.extractor.trafilatura")
    def test_download_failure(self, mock_traf):
        mock_traf.fetch_url.return_value = None
        result = extract_article_text("https://example.com")
        assert result is None

    @patch("signalstack.processing.extractor.trafilatura")
    def test_extraction_returns_none(self, mock_traf):
        mock_traf.fetch_url.return_value = "<html>page</html>"
        mock_traf.extract.return_value = None
        result = extract_article_text("https://example.com")
        assert result is None

    @patch("signalstack.processing.extractor.trafilatura")
    def test_content_too_short(self, mock_traf):
        mock_traf.fetch_url.return_value = "<html>page</html>"
        mock_traf.extract.return_value = "Short"
        result = extract_article_text("https://example.com", min_content_length=300)
        assert result is None

    @patch("signalstack.processing.extractor.trafilatura")
    def test_binary_content_rejected(self, mock_traf):
        mock_traf.fetch_url.return_value = "<html>page</html>"
        mock_traf.extract.return_value = "\x00\x01\x02" * 200
        result = extract_article_text("https://example.com")
        assert result is None

    @patch("signalstack.processing.extractor.trafilatura")
    def test_successful_extraction(self, mock_traf):
        content = "This is a valid article. " * 50
        mock_traf.fetch_url.return_value = "<html>page</html>"
        mock_traf.extract.return_value = content
        result = extract_article_text("https://example.com")
        assert result == content.strip()

    @patch("signalstack.processing.extractor.trafilatura")
    def test_custom_min_content_length(self, mock_traf):
        content = "x" * 100
        mock_traf.fetch_url.return_value = "<html>page</html>"
        mock_traf.extract.return_value = content
        assert extract_article_text("https://example.com", min_content_length=50) is not None
        assert extract_article_text("https://example.com", min_content_length=200) is None

    @patch("signalstack.processing.extractor.trafilatura")
    def test_download_exception(self, mock_traf):
        mock_traf.fetch_url.side_effect = Exception("Network error")
        result = extract_article_text("https://example.com")
        assert result is None

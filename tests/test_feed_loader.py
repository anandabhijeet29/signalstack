import pytest

from signalstack.ingestion.feed_loader import load_feeds


class TestLoadFeeds:
    def test_valid_yaml(self, tmp_path):
        path = tmp_path / "feeds.yaml"
        path.write_text("feeds:\n  - https://example.com/feed\n  - https://other.com/feed\n")
        result = load_feeds(str(path))
        assert result == ["https://example.com/feed", "https://other.com/feed"]

    def test_missing_feeds_key(self, tmp_path):
        path = tmp_path / "feeds.yaml"
        path.write_text("other_key: value\n")
        result = load_feeds(str(path))
        assert result == []

    def test_non_list_feeds(self, tmp_path):
        path = tmp_path / "feeds.yaml"
        path.write_text("feeds: not_a_list\n")
        with pytest.raises(ValueError, match="list"):
            load_feeds(str(path))

    def test_non_string_entries(self, tmp_path):
        path = tmp_path / "feeds.yaml"
        path.write_text("feeds:\n  - 123\n  - true\n")
        with pytest.raises(ValueError, match="strings"):
            load_feeds(str(path))

    def test_empty_file(self, tmp_path):
        path = tmp_path / "feeds.yaml"
        path.write_text("")
        result = load_feeds(str(path))
        assert result == []

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_feeds(str(tmp_path / "nonexistent.yaml"))

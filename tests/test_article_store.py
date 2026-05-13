import datetime
import json

import pytest

from signalstack.models.article import Article
from signalstack.processing.article_store import (
    filter_new_articles,
    load_seen_urls,
    save_seen_urls,
    update_seen_urls,
)


def _make_article(title: str, link: str) -> Article:
    return Article(title=title, link=link, source="Test")


class TestLoadSeenUrls:
    def test_missing_file(self, tmp_path):
        result = load_seen_urls(str(tmp_path / "nonexistent.json"))
        assert result == {}

    def test_corrupted_json(self, tmp_path):
        path = tmp_path / "seen.json"
        path.write_text("not json", encoding="utf-8")
        result = load_seen_urls(str(path))
        assert result == {}

    def test_valid_file(self, tmp_path):
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        data = {"seen_urls": {"https://example.com": now}}
        path = tmp_path / "seen.json"
        path.write_text(json.dumps(data), encoding="utf-8")

        result = load_seen_urls(str(path), max_age_days=7)
        assert "https://example.com" in result

    def test_expired_urls_removed(self, tmp_path):
        old_time = (
            datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=10)
        ).isoformat()
        data = {"seen_urls": {"https://old.com": old_time}}
        path = tmp_path / "seen.json"
        path.write_text(json.dumps(data), encoding="utf-8")

        result = load_seen_urls(str(path), max_age_days=7)
        assert "https://old.com" not in result

    def test_invalid_timestamp_skipped(self, tmp_path):
        data = {"seen_urls": {"https://bad.com": "not-a-date"}}
        path = tmp_path / "seen.json"
        path.write_text(json.dumps(data), encoding="utf-8")

        result = load_seen_urls(str(path))
        assert "https://bad.com" not in result

    def test_naive_timestamp_treated_as_utc(self, tmp_path):
        naive = datetime.datetime.now().replace(microsecond=0).isoformat()
        data = {"seen_urls": {"https://naive.com": naive}}
        path = tmp_path / "seen.json"
        path.write_text(json.dumps(data), encoding="utf-8")

        result = load_seen_urls(str(path), max_age_days=7)
        assert "https://naive.com" in result


class TestSaveSeenUrls:
    def test_creates_file(self, tmp_path):
        path = tmp_path / "subdir" / "seen.json"
        save_seen_urls(str(path), {"https://example.com": "2026-01-01T00:00:00"})
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert "https://example.com" in data["seen_urls"]


class TestFilterNewArticles:
    def test_all_new(self):
        articles = [_make_article("A", "https://a.com"), _make_article("B", "https://b.com")]
        result = filter_new_articles(articles, {})
        assert len(result) == 2

    def test_all_seen(self):
        articles = [_make_article("A", "https://a.com")]
        seen = {"https://a.com": "2026-01-01T00:00:00"}
        result = filter_new_articles(articles, seen)
        assert len(result) == 0

    def test_mixed(self):
        articles = [
            _make_article("A", "https://a.com"),
            _make_article("B", "https://b.com"),
        ]
        seen = {"https://a.com": "2026-01-01T00:00:00"}
        result = filter_new_articles(articles, seen)
        assert len(result) == 1
        assert result[0].link == "https://b.com"

    def test_empty_link_filtered(self):
        articles = [_make_article("A", ""), _make_article("B", "https://b.com")]
        result = filter_new_articles(articles, {})
        assert len(result) == 1

    def test_deduplicates_within_batch(self):
        articles = [
            _make_article("A", "https://same.com"),
            _make_article("B", "https://same.com"),
        ]
        result = filter_new_articles(articles, {})
        assert len(result) == 1


class TestUpdateSeenUrls:
    def test_adds_new_urls(self):
        articles = [_make_article("A", "https://new.com")]
        result = update_seen_urls(articles, {})
        assert "https://new.com" in result

    def test_preserves_existing(self):
        articles = [_make_article("A", "https://new.com")]
        existing = {"https://old.com": "2026-01-01T00:00:00"}
        result = update_seen_urls(articles, existing)
        assert "https://old.com" in result
        assert "https://new.com" in result

    def test_does_not_overwrite_existing(self):
        articles = [_make_article("A", "https://existing.com")]
        existing = {"https://existing.com": "2026-01-01T00:00:00"}
        result = update_seen_urls(articles, existing)
        assert result["https://existing.com"] == "2026-01-01T00:00:00"

    def test_timestamp_is_utc(self):
        articles = [_make_article("A", "https://tz.com")]
        result = update_seen_urls(articles, {})
        ts = result["https://tz.com"]
        parsed = datetime.datetime.fromisoformat(ts)
        assert parsed.tzinfo is not None

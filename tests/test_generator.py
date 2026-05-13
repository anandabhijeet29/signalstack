from signalstack.digest.generator import generate_digest, save_digest


def _make_summary(title: str, importance: int = 5) -> dict:
    return {
        "title": title,
        "source": "TestSource",
        "url": f"https://example.com/{title.lower().replace(' ', '-')}",
        "reading_time": 3,
        "importance_score": importance,
        "tldr": f"TLDR for {title}",
        "summary": f"Summary for {title}",
        "key_insights": [f"Insight 1 for {title}", f"Insight 2 for {title}"],
    }


class TestGenerateDigest:
    def test_basic_structure(self):
        summaries = [_make_summary("Test Article")]
        result = generate_digest(summaries)
        assert "# SignalStack Weekly Intelligence" in result
        assert "### Test Article" in result
        assert "TLDR for Test Article" in result
        assert "Insight 1 for Test Article" in result

    def test_with_themes(self):
        summaries = [_make_summary("Test")]
        themes = ["AI is growing", "Compute costs are dropping"]
        result = generate_digest(summaries, themes=themes)
        assert "## Major Themes This Week" in result
        assert "AI is growing" in result
        assert "Compute costs are dropping" in result

    def test_without_themes(self):
        summaries = [_make_summary("Test")]
        result = generate_digest(summaries, themes=[])
        assert "## Major Themes" not in result

    def test_no_dangling_top_signals_header(self):
        summaries = [_make_summary("Test")]
        result = generate_digest(summaries, themes=["Theme 1"])
        assert "## Top Signals" not in result

    def test_sorted_by_importance_descending(self):
        summaries = [
            _make_summary("Low", importance=2),
            _make_summary("High", importance=9),
            _make_summary("Mid", importance=5),
        ]
        result = generate_digest(summaries)
        high_pos = result.index("### High")
        mid_pos = result.index("### Mid")
        low_pos = result.index("### Low")
        assert high_pos < mid_pos < low_pos

    def test_frontmatter_present(self):
        result = generate_digest([_make_summary("Test")])
        assert result.startswith("---\n")
        assert "title: SignalStack Weekly Intelligence" in result
        assert "tags: [signalstack, ai, intelligence]" in result

    def test_empty_summaries(self):
        result = generate_digest([])
        assert "# SignalStack Weekly Intelligence" in result


class TestSaveDigest:
    def test_creates_file(self, tmp_path):
        save_digest("# Test", str(tmp_path))
        files = list(tmp_path.glob("signalstack_weekly_*.md"))
        assert len(files) == 1
        assert files[0].read_text(encoding="utf-8") == "# Test"

    def test_creates_nested_directory(self, tmp_path):
        nested = tmp_path / "deep" / "nested"
        save_digest("# Test", str(nested))
        files = list(nested.glob("signalstack_weekly_*.md"))
        assert len(files) == 1

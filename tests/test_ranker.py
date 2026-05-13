from signalstack.agents.ranker import _score_article, rank_articles
from signalstack.models.article import Article


def _make_article(title: str, preview: str = "") -> Article:
    return Article(title=title, link=f"https://example.com/{title}", source="Test", preview=preview)


class TestScoreArticle:
    def test_keyword_hits_increase_score(self):
        no_keywords = _make_article("Breaking news today")
        one_keyword = _make_article("Breaking AI news today")
        two_keywords = _make_article("AI model news today")

        assert _score_article(one_keyword) > _score_article(no_keywords)
        assert _score_article(two_keywords) > _score_article(one_keyword)

    def test_all_keywords_score(self):
        article = _make_article("ai model agent compute startup policy research")
        score = _score_article(article)
        assert score >= 7 * 3.0

    def test_title_length_contributes_to_score(self):
        short = _make_article("AI")
        long = _make_article("AI research on new model architectures for agents")
        assert _score_article(long) > _score_article(short)

    def test_title_length_capped_at_120(self):
        at_cap = _make_article("x" * 120)
        over_cap = _make_article("x" * 200)
        assert _score_article(at_cap) == _score_article(over_cap)

    def test_keywords_in_preview_count(self):
        no_preview = _make_article("News today", preview="")
        with_preview = _make_article("News today", preview="this is about AI research")
        assert _score_article(with_preview) > _score_article(no_preview)

    def test_keyword_matching_is_word_boundary(self):
        article = _make_article("The aim is to train")
        score = _score_article(article)
        no_kw = _make_article("x" * len("The aim is to train"))
        assert score == _score_article(no_kw)


class TestRankArticles:
    def test_returns_top_n(self):
        articles = [_make_article(f"Article {i}") for i in range(10)]
        result = rank_articles(articles, top_n=3)
        assert len(result) == 3

    def test_empty_list(self):
        assert rank_articles([], top_n=5) == []

    def test_top_n_zero(self):
        articles = [_make_article("Test")]
        assert rank_articles(articles, top_n=0) == []

    def test_top_n_negative(self):
        articles = [_make_article("Test")]
        assert rank_articles(articles, top_n=-1) == []

    def test_top_n_exceeds_list_length(self):
        articles = [_make_article(f"Article {i}") for i in range(3)]
        result = rank_articles(articles, top_n=10)
        assert len(result) == 3

    def test_higher_scoring_articles_rank_first(self):
        low = _make_article("Short title")
        high = _make_article("AI model research on compute for agents")
        result = rank_articles([low, high], top_n=2)
        assert result[0].title == high.title

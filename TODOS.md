# TODOS

## LLM response schema validation
**What:** Add validation (pydantic or dict checks) for JSON responses from OpenAI in summarizer.py and synthesizer.py.
**Why:** Bare `json.loads()` with direct key access crashes with KeyError if the LLM returns unexpected structure. Graceful validation would log a warning and skip malformed responses.
**Where:** `signalstack/agents/summarizer.py:90`, `signalstack/agents/synthesizer.py:62`
**Depends on:** None

## Simplify two-pass ranking
**What:** The pipeline ranks articles twice (once with RSS summaries, once with extracted previews). The keyword-based scorer produces near-identical results in both passes. Simplify to a single pass: extract all new articles, then rank once with full content.
**Why:** The two-pass design implies sophistication the scoring function can't deliver. One pass would produce nearly identical results with simpler code. Alternatively, make the second pass actually leverage richer content (e.g., LLM-based relevance).
**Where:** `signalstack/pipeline.py:59-81`, `signalstack/agents/ranker.py`
**Depends on:** None

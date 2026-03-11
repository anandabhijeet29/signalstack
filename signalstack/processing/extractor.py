from typing import Optional

import trafilatura


def looks_like_text(text: str) -> bool:
    if not text:
        return False

    printable_chars = sum(c.isprintable() for c in text)
    ratio = printable_chars / len(text)
    return ratio > 0.85


def extract_article_text(url: str) -> Optional[str]:
    print(f"Extracting article: {url}")

    try:
        downloaded = trafilatura.fetch_url(url)
    except Exception:
        print("Failed to download page")
        return None

    if not downloaded:
        print("Failed to download page")
        return None

    try:
        text = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=False,
        )
    except Exception:
        print("Extraction returned no content")
        return None

    if text is None:
        print("Extraction returned no content")
        return None

    print("Running text sanity check...")
    if not looks_like_text(text):
        print(f"Corrupted or binary content detected: {url}")
        return None

    cleaned_text = text.strip()
    if len(cleaned_text) < 300:
        print("Content too short — likely paywalled or preview-only")
        return None

    print("Extraction successful")
    print(f"Extracted length: {len(cleaned_text)} characters")
    return cleaned_text

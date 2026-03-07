from typing import Optional

import trafilatura


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

    cleaned_text = text.strip()
    if len(cleaned_text) < 300:
        print("Content too short — likely paywalled or preview-only")
        return None

    print("Extraction successful")
    print(f"Extracted length: {len(cleaned_text)} characters")
    return cleaned_text

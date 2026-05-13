from dataclasses import dataclass
from typing import Optional


@dataclass
class Article:
    title: str
    link: str
    source: str
    summary: Optional[str] = None
    preview: Optional[str] = None
    content: Optional[str] = None

    def has_content(self, min_length: int = 300) -> bool:
        return bool(self.content and len(self.content.strip()) > min_length)

    def short_title(self) -> str:
        return self.title[:80]

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

    def has_content(self) -> bool:
        return bool(self.content and len(self.content.strip()) > 300)

    def short_title(self) -> str:
        return self.title[:80]

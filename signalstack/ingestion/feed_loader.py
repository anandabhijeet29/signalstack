from typing import List
import yaml


def load_feeds(path: str) -> List[str]:
    """Load RSS feed URLs from a YAML file under the `feeds` key."""
    with open(path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    feeds = data.get("feeds", [])
    if not isinstance(feeds, list):
        raise ValueError("Expected `feeds` to be a list in YAML.")

    if not all(isinstance(feed, str) for feed in feeds):
        raise ValueError("All entries in `feeds` must be strings.")

    return feeds

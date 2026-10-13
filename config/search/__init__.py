"""Web search dispatcher.

Default route is SearXNG (local, free). When "Search Pro" is enabled per request
(web_search_pro=True) the primary becomes Tavily. The other provider is always the
fallback (used if the primary errors or returns nothing).
"""
from dotenv import load_dotenv

load_dotenv()  # ensure env is populated before the provider modules read it

from . import searxng, tavily  # noqa: E402


def web_search(query: str, max_results: int = 5, pro: bool = False):
    """Search via the chosen provider, falling back to the other on failure/empty.

    pro=False (default) -> SearXNG primary, Tavily fallback.
    pro=True ("Search Pro") -> Tavily primary, SearXNG fallback.
    """
    order = (("tavily", tavily), ("searxng", searxng)) if pro else (("searxng", searxng), ("tavily", tavily))
    for name, provider in order:
        try:
            results = provider.search(query, max_results)
            if results:
                for r in results:
                    r["provider"] = name
                return results
        except Exception:
            continue
    return []


def search_health() -> dict:
    """Per-provider status for /health."""
    return {"searxng": searxng.health(), "tavily": tavily.health()}

"""Tavily web-search provider (the "Search Pro" route).

General web with richer extracted content than SearXNG snippets. Requires
TAVILY_API_KEY; returns [] (and reports not_configured) when it's unset.
"""
import os


def search(query: str, max_results: int = 5):
    """Return [{'url','title','content'}] from Tavily, or [] if no key."""
    key = os.getenv("TAVILY_API_KEY")
    if not key:
        return []
    from tavily import TavilyClient

    res = TavilyClient(api_key=key).search(query, max_results=max_results)
    return [
        {"url": d.get("url", ""), "title": d.get("title", ""), "content": d.get("content", "")}
        for d in res.get("results", [])
    ]


def health() -> dict:
    key = os.getenv("TAVILY_API_KEY")
    if not key:
        return {"status": "not_configured"}
    try:
        search("health", 1)
        return {"status": "up"}
    except Exception:
        return {"status": "down"}

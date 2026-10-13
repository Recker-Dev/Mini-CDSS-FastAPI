"""SearXNG web-search provider.

Defaults to the medical engines (pubmed + crossref), which return clean, clinical
results — Bing poisons disease queries with common-word prefixes (e.g. "Type 2
Diabetes" -> typing sites). Add engines via SEARXNG_ENGINES if you want general web.
"""
import os
from urllib.parse import urlparse

SEARXNG_URL = os.getenv("SEARXNG_URL", "http://localhost:32768").rstrip("/")
ENGINES = os.getenv("SEARXNG_ENGINES", "pubmed,crossref")

# SERP noise some general engines surface for the lead term.
_DENY_DOMAINS = {
    "dictionary.cambridge.org", "merriam-webster.com", "thefreedictionary.com",
    "dictionary.com", "vocabulary.com", "collinsdictionary.com", "wordnik.com",
    "typing.com", "typingclub.com", "typingtest.com", "monkeytype.com",
}


def _denied(url: str) -> bool:
    host = urlparse(url).netloc.lower()
    host = host[4:] if host.startswith("www.") else host
    return any(host == d or host.endswith("." + d) for d in _DENY_DOMAINS)


def search(query: str, max_results: int = 5):
    """Return [{'url','title','content'}] from SearXNG."""
    import requests

    resp = requests.get(
        f"{SEARXNG_URL}/search",
        params={"q": query, "format": "json", "engines": ENGINES},
        timeout=20,
    )
    resp.raise_for_status()
    out = []
    for r in resp.json().get("results", []):
        url = r.get("url", "")
        content = (r.get("content") or "").strip()
        if not url or not content or _denied(url):
            continue
        out.append({"url": url, "title": r.get("title", ""), "content": content})
        if len(out) >= max_results:
            break
    return out


def health() -> dict:
    """Reachability of the SearXNG instance."""
    try:
        import requests

        r = requests.get(
            f"{SEARXNG_URL}/search",
            params={"q": "health", "format": "json", "engines": ENGINES},
            timeout=15,
        )
        return {"status": "up" if r.status_code == 200 else "down"}
    except Exception:
        return {"status": "down"}

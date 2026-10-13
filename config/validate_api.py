"""Health checks for the services this app depends on.

The project no longer uses Gemini or Groq. The only hard dependency is the local
LLM; Tavily (web search) is optional.
"""
import os
import time

from .llm import chat, MODEL


def validate_local_llm() -> str:
    """Return 'ValidAPI' if the local LLM responds, else 'InvalidAPI'."""
    try:
        chat([{"role": "user", "content": "Reply with the single word OK."}])
        return "ValidAPI"
    except Exception:
        return "InvalidAPI"


def validate_tavily(tav_api: str) -> str:
    """Tavily is optional; treat a missing key as acceptable."""
    return "ValidAPI"


def validate_keys(tav_api=None, gemini_api=None, groq_api=None) -> str:
    """Kept for backwards-compatible endpoints.

    Only the local LLM is actually required now; the gemini/groq arguments are
    ignored and accepted only so existing callers don't break.
    """
    if validate_local_llm() == "ValidAPI":
        return "Validated"
    return "Local LLM is not reachable. Check LOCAL_LLM_BASE_URL and that the server is running."


# ---- Structured health report (for /health) -------------------------------
def check_llm() -> dict:
    """Live check of the local LLM. Error details are kept out of the response
    to avoid leaking internal origins on a publicly-served health page."""
    started = time.time()
    try:
        chat([{"role": "user", "content": "Reply with the single word OK."}])
        return {"status": "up", "model": MODEL, "latency_ms": round((time.time() - started) * 1000)}
    except Exception:
        return {"status": "down", "model": MODEL}


def health_report() -> dict:
    """Aggregate status for the backend, the local LLM, and web search."""
    from .search import search_health

    return {
        "backend": "ok",
        "llm": check_llm(),
        "search": search_health(),
    }

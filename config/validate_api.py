"""Health checks for the services this app depends on.

The project no longer uses Gemini or Groq. The only hard dependency is the local
LLM; Tavily (web search) is optional.
"""
import os

from .llm import chat


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

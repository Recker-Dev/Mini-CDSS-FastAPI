"""Shared access to the local LLM and the local embedding model.

Everything in this project that needs text generation goes through the official
OpenAI Python client, pointed at a local OpenAI-compatible server (llama.cpp /
Ollama / vLLM, etc.). Nothing here calls out to a hosted provider.

Configure via environment variables (see .env):
    LOCAL_LLM_BASE_URL   required, e.g. http://localhost:55555/v1
    LOCAL_LLM_MODEL      required, e.g. qwen3.5
    LOCAL_LLM_API_KEY    optional, defaults to "local" (most servers ignore it)
"""
import os
from functools import lru_cache
from typing import List, Type, TypeVar

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

# This module is imported (transitively) before main.py calls load_dotenv(), so
# load the .env here to make sure the configuration below is populated.
load_dotenv()

BASE_URL = os.getenv("LOCAL_LLM_BASE_URL")
MODEL = os.getenv("LOCAL_LLM_MODEL")
# Most local servers ignore the key; it only needs to be non-empty.
API_KEY = os.getenv("LOCAL_LLM_API_KEY", "local")
# Safety ceilings on generated tokens (NOT a tight cap — natural outputs are far
# smaller; this only guards against runaway generation). Prose reports top out
# around ~750 tokens, so 2048 never truncates them. Structured JSON can be larger
# and varies with inputs (e.g. diagnosis_count), and a truncated JSON is fatal,
# so it gets a bigger ceiling plus an explicit truncation check below.
MAX_TOKENS = int(os.getenv("LOCAL_LLM_MAX_TOKENS", "3072"))
STRUCTURED_MAX_TOKENS = int(os.getenv("LOCAL_LLM_STRUCTURED_MAX_TOKENS", "4096"))

if not BASE_URL or not MODEL:
    raise RuntimeError(
        "LOCAL_LLM_BASE_URL and LOCAL_LLM_MODEL must be set (see .env). "
        "Example: LOCAL_LLM_BASE_URL=http://localhost:55555/v1, LOCAL_LLM_MODEL=qwen3.5"
    )

# Qwen3.5 (and other hybrid-reasoning models) expose a "thinking" mode. Kept off
# by default: it's verbose, slow, and breaks grammar-constrained structured output
# on an 8k slot. Toggle via LOCAL_LLM_ENABLE_THINKING if you ever want it globally.
ENABLE_THINKING = os.getenv("LOCAL_LLM_ENABLE_THINKING", "false").strip().lower() in ("1", "true", "yes", "on")
_EXTRA_BODY = {"chat_template_kwargs": {"enable_thinking": ENABLE_THINKING}}

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

Message = dict  # {"role": "system"|"user"|"assistant", "content": str}
T = TypeVar("T", bound=BaseModel)

# Sentinel so callers can distinguish "use the default ceiling" from an explicit
# `None` meaning "no cap at all" (used by the elaborative report builders).
_DEFAULT = object()


def _resolve_max_tokens(max_tokens):
    """Return kwargs for the API call: {} means uncapped (model/context-bound)."""
    if max_tokens is _DEFAULT:
        return {"max_tokens": MAX_TOKENS}
    if max_tokens is None:
        return {}  # uncapped
    return {"max_tokens": max_tokens}


def sys(content: str) -> Message:
    """Build a system message."""
    return {"role": "system", "content": content}


def user(content: str) -> Message:
    """Build a user message."""
    return {"role": "user", "content": content}


def chat(messages: List[Message], temperature: float = 0, max_tokens=_DEFAULT, **kwargs) -> str:
    """Run a chat completion and return the plain text content.

    max_tokens: omit for the default ceiling, pass None for uncapped, or an int.
    """
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=temperature,
        extra_body=_EXTRA_BODY,
        **_resolve_max_tokens(max_tokens),
        **kwargs,
    )
    return resp.choices[0].message.content


def chat_stream(messages: List[Message], temperature: float = 0, max_tokens=_DEFAULT):
    """Stream a chat completion, yielding text chunks as they are generated.

    Used by the report/answer endpoints so output appears token-by-token instead
    of after the whole generation finishes. Pass max_tokens=None for uncapped
    (used by the elaborative report builders).
    """
    stream = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=temperature,
        extra_body=_EXTRA_BODY,
        stream=True,
        **_resolve_max_tokens(max_tokens),
    )
    for chunk in stream:
        if not chunk.choices:
            continue
        choice = chunk.choices[0]
        delta = choice.delta.content
        if delta:
            yield delta
        # Don't let output silently cut off mid-sentence — say so if it does.
        if choice.finish_reason == "length":
            yield (
                "\n\n*[Output truncated at the "
                f"{MAX_TOKENS}-token limit — raise LOCAL_LLM_MAX_TOKENS in .env.]*"
            )


def structured(messages: List[Message], schema: Type[T], temperature: float = 0) -> T:
    """Run a chat completion constrained to a Pydantic schema and return an instance.

    Uses the server's json_schema response format (grammar-constrained decoding),
    so the output is guaranteed to parse into ``schema``.
    """
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=STRUCTURED_MAX_TOKENS,
        extra_body=_EXTRA_BODY,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema()},
        },
    )
    choice = resp.choices[0]
    # A truncated structured response yields invalid JSON; fail loudly and clearly
    # rather than with a cryptic parse error.
    if choice.finish_reason == "length":
        raise RuntimeError(
            f"{schema.__name__} output hit the {STRUCTURED_MAX_TOKENS}-token ceiling and was "
            "truncated. Increase LOCAL_LLM_STRUCTURED_MAX_TOKENS in .env."
        )
    return schema.model_validate_json(choice.message.content)


@lru_cache(maxsize=1)
def get_embedding_function():
    """Return a process-wide shared local embedding function (all-MiniLM-L6-v2).

    Loaded once and cached: creating it per request would reload the model from
    disk every time, which is what made the old Gemini-embedding path feel slow.
    Runs fully locally (ONNX, CPU), no API key required.
    """
    import chromadb.utils.embedding_functions as embedding_functions

    return embedding_functions.DefaultEmbeddingFunction()

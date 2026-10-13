"""Load system prompts from markdown files in the ``prompts/`` folder.

Prompts live outside the code so they can be edited without touching Python.
Templated prompts keep their ``{placeholder}`` tokens and are ``.format(...)``-ed
by the caller as before.

Override the location with the PROMPTS_DIR env var if needed.
"""
import os
from functools import lru_cache
from pathlib import Path

PROMPTS_DIR = Path(
    os.getenv("PROMPTS_DIR", str(Path(__file__).resolve().parent.parent / "prompts"))
)


@lru_cache(maxsize=None)
def load_prompt(name: str) -> str:
    """Return the text of ``prompts/<name>.md`` (cached), stripped of outer whitespace."""
    path = PROMPTS_DIR / f"{name}.md"
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Prompt '{name}' not found at {path}") from exc

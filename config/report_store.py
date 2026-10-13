"""In-memory capture of generated report text, per thread.

As each report streams to the client it is also accumulated here, so the PDF can
be assembled from exactly what was produced — no second LLM pass. Process-local
and ephemeral (same lifetime model as the vector DBs); fine for this app.
"""
from typing import Dict

# thread_id -> {"ner": str, "prelim": str, "bestprac": str}
_store: Dict[str, Dict[str, str]] = {}


def set_report(thread_id: str, key: str, text: str) -> None:
    _store.setdefault(thread_id, {})[key] = text


def get_reports(thread_id: str) -> Dict[str, str]:
    return _store.get(thread_id, {})

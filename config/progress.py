"""Turn LangGraph node updates into frontend-friendly progress events.

Each graph-run endpoint streams Server-Sent Events of the form:

    data: {"label": "...", "detail": "..."}\\n\\n

`label` is a short step headline; `detail` (optional) is the surfaced result for
the nodes we choose to expose. Returning None skips a node (e.g. no-op/interrupt).
"""
import json


def sse(event: dict) -> str:
    """Encode one progress event as an SSE `data:` frame."""
    return f"data: {json.dumps(event)}\n\n"


def main_graph_event(node_name: str, update):
    """Map a main-graph node update to a progress event (or None to skip)."""
    update = update or {}

    if node_name == "perform and validate NER extractions":
        return {"label": "🔬 Extracting & validating clinical entities…"}

    if node_name == "perform preliminary diagnosis node":
        return {"label": "🩺 Forming preliminary diagnoses…"}

    if node_name == "extract diseases":
        diseases = update.get("diseases", []) or []
        detail = "\n".join(f"- {d}" for d in diseases) or None
        return {"label": "🏷️ Identified conditions", "detail": detail}

    if node_name == "search web for best practises":
        # Each Send fan-out surfaces one disease's sources. Include the RAW
        # retrieved results (url + snippet) — the data the report LLM later
        # formats — so the reasoning panel shows real evidence.
        best_practises = update.get("best_practises", []) or []
        if best_practises:
            disease = best_practises[0].get("disease", "condition")
            n = len(best_practises)
            lines = []
            for b in best_practises:
                url = b.get("url", "")
                snippet = (b.get("content", "") or "").strip().replace("\n", " ")
                if not url:
                    continue
                snippet = (snippet[:140] + "…") if len(snippet) > 140 else snippet
                lines.append(f"- [{url}]({url})" + (f"\n  {snippet}" if snippet else ""))
            detail = "\n".join(lines) or None
            return {"label": f"🌐 {disease}: {n} source(s)", "detail": detail}
        return {"label": "🌐 Searching best practices…"}

    # No-op interrupt node / interrupt marker: nothing useful to show.
    if node_name in ("prelim human feedback node", "__interrupt__"):
        return None

    return {"label": f"⚙️ {node_name}"}


def rag_event(node_name: str, update):
    """Map a RAG-graph node update to a progress event (or None to skip)."""
    update = update or {}

    if node_name == "frame queries":
        detail = None
        queries = update.get("queries")
        if queries is not None:
            try:
                detail = "\n".join(f"- {entry.query}" for entry in queries.queries) or None
            except Exception:
                detail = None
        return {"label": "✍️ Framed search queries", "detail": detail}

    if node_name == "retrieve docs":
        docs = update.get("docs_retrieved") or []
        return {"label": "📚 Retrieved relevant document chunks", "detail": f"- {len(docs)} query result set(s)"}

    if node_name == "check relevance":
        rel = update.get("relevance")
        return {"label": "✅ Relevance check", "detail": f"- Relevant: {rel}"}

    return {"label": f"⚙️ {node_name}"}

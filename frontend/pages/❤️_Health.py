import os
import html
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(page_title="Mini-CDSS Health", page_icon="❤️")
st.title("System Health")
st.caption(f"Backend: `{BACKEND_URL}`")

if st.button("🔄 Refresh"):
    st.rerun()

# Fetch the aggregate health report from the backend.
try:
    resp = requests.get(f"{BACKEND_URL}/health", timeout=40)
    resp.raise_for_status()
    data = resp.json()
except Exception as e:
    st.error(f"Cannot reach the backend at {BACKEND_URL}\n\n{e}")
    st.stop()

# status -> (label, colors). Green (dimmed) = active, red = inactive, grey = n/a.
_STYLES = {
    "active":         ("Active",         "#2f9e5f", "rgba(47,158,95,0.12)",  "rgba(47,158,95,0.35)"),
    "inactive":       ("Inactive",       "#c0392b", "rgba(192,57,43,0.10)",  "rgba(192,57,43,0.32)"),
    "not_configured": ("Not configured", "#8a857b", "rgba(138,133,123,0.10)", "rgba(138,133,123,0.28)"),
}


def _state(status: str) -> str:
    return {"up": "active", "ok": "active", "down": "inactive"}.get(status, status if status in _STYLES else "not_configured")


def card(name: str, status: str, note: str = "") -> str:
    label, fg, bg, border = _STYLES[_state(status)]
    note_html = f'<div style="color:#8a857b;font-size:0.8rem;margin-top:2px;">{html.escape(note)}</div>' if note else ""
    return (
        f'<div style="display:flex;justify-content:space-between;align-items:center;'
        f'padding:12px 16px;margin-bottom:10px;border-radius:10px;'
        f'background:{bg};border:1px solid {border};">'
        f'<div><div style="font-weight:600;color:#2b2a27;">{html.escape(name)}</div>{note_html}</div>'
        f'<div style="display:flex;align-items:center;gap:8px;color:{fg};font-weight:600;font-size:0.9rem;">'
        f'<span style="width:9px;height:9px;border-radius:50%;background:{fg};display:inline-block;"></span>'
        f'{label}</div></div>'
    )


llm = data.get("llm", {})
search = data.get("search", {})
sx = search.get("searxng", {})
tv = search.get("tavily", {})

llm_note = ""
if llm.get("model"):
    llm_note = llm["model"] + (f" · {llm['latency_ms']} ms" if llm.get("latency_ms") is not None else "")

cards = [
    card("Backend API", data.get("backend", "down")),
    card("Local LLM", llm.get("status", "down"), llm_note),
    card("Web Search · SearXNG", sx.get("status", "down"), "Default route (local)"),
    card("Web Search · Tavily", tv.get("status", "not_configured"), "Search Pro route (fallback)"),
]
st.markdown("".join(cards), unsafe_allow_html=True)

core_ok = data.get("backend") == "ok" and llm.get("status") == "up"
st.divider()
if core_ok:
    st.success("Core services healthy.")
else:
    st.error("One or more core services are unavailable.")

with st.expander("Raw report"):
    st.json(data)

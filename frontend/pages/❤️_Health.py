import os
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(page_title="Mini-CDSS Health", page_icon="❤️")
st.title("❤️ System Health")
st.caption(f"Backend: `{BACKEND_URL}`")

if st.button("🔄 Refresh"):
    st.rerun()

# Fetch the aggregate health report from the backend.
try:
    resp = requests.get(f"{BACKEND_URL}/health", timeout=30)
    resp.raise_for_status()
    data = resp.json()
except Exception as e:
    st.error(f"🔴 Cannot reach the backend at {BACKEND_URL}\n\n{e}")
    st.stop()

llm = data.get("llm", {})
tavily = data.get("tavily", {})

backend_ok = data.get("backend") == "ok"
llm_up = llm.get("status") == "up"

col1, col2, col3 = st.columns(3)

# 1. Backend
col1.metric("Backend API", "🟢 Online" if backend_ok else "🔴 Offline")

# 2. Local LLM
llm_latency = f"{llm.get('latency_ms')} ms" if llm.get("latency_ms") is not None else None
col2.metric(
    "Local LLM",
    "🟢 Up" if llm_up else "🔴 Down",
    delta=llm.get("model"),
    delta_color="off",
)
if llm_latency:
    col2.caption(f"Latency: {llm_latency}")

# 3. Tavily Web Search
tavily_labels = {
    "up": "🟢 Up",
    "down": "🔴 Down",
    "not_configured": "⚪ Not configured",
}
col3.metric("Tavily Web Search", tavily_labels.get(tavily.get("status"), "❔ Unknown"))
if tavily.get("status") == "not_configured":
    col3.caption("Web search is optional — set TAVILY_API_KEY to enable.")

st.divider()

# Overall banner (Tavily is optional, so it doesn't fail the overall health).
if backend_ok and llm_up:
    st.success("All core services are healthy.")
else:
    st.error("One or more core services are unavailable.")

with st.expander("Raw report"):
    st.json(data)

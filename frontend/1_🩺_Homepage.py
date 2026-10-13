import streamlit as st
import uuid
import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

thread = {"configurable": {"thread_id":"1"}}
persist_directory =".chroma"

# Backend origin is env-driven (see .env) — switch local/hosted in one place.
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")

FASTAPI_URL_VALIDATE_API = f"{BACKEND_URL}/validate_and_set_api"
FASTAPI_URL_SETTING_API = f"{BACKEND_URL}/set_api"
FASTAPI_URL_GRAPHSTART = f"{BACKEND_URL}/graphstart/"
FASTAPI_URL_PRELIM_INTERRUPT = f"{BACKEND_URL}/prelimInterruptTrigger"
FASTAPI_URL_NER_REPORT = f"{BACKEND_URL}/nerReport"
FASTAPI_URL_PRELIM_REPORT = f"{BACKEND_URL}/prelimReport"
FASTAPI_URL_BESTPRAC_REPORT = f"{BACKEND_URL}/bestpracReport"
FASTAPI_URL_CREATE_VECTOR_DB = f"{BACKEND_URL}/addFilesAndCreateVectorDB"
FASTAPI_URL_RAG_SEARCH = f"{BACKEND_URL}/ragSearch"
FASTAPI_URL_RAG_ANSWER = f"{BACKEND_URL}/ragAnswer"
FASTAPI_URL_EXTRACT_MEDICAL_TRIGGER = f"{BACKEND_URL}/extractMedicalDetails"
FASTAPI_URL_EXTRACT_MEDICAL_DETAILS = f"{BACKEND_URL}/medicalInsightReport"
FASTAPI_URL_REPORT_PDF = f"{BACKEND_URL}/reportPdf"


def fetch_report_pdf(thread_id: str):
    """Download the backend-rendered report PDF (bytes), or None on failure."""
    try:
        resp = requests.get(FASTAPI_URL_REPORT_PDF, params={"thread_id": thread_id}, timeout=60)
        if resp.status_code == 200:
            return resp.content
    except Exception:
        pass
    return None


def stream_report(url: str, payload: dict):
    """Yield text chunks from a streaming backend endpoint (for st.write_stream)."""
    with requests.post(url, json=payload, stream=True) as response:
        if response.status_code != 200:
            yield f"Error {response.status_code}: {response.text}"
            return
        for chunk in response.iter_content(chunk_size=None):
            if chunk:
                yield chunk.decode("utf-8", errors="ignore")


def stream_final_reports(thread_id: str):
    """Chain the NER, prelim and best-practices reports into one streamed message."""
    payload = {"thread_id": thread_id}
    yield from stream_report(FASTAPI_URL_NER_REPORT, payload)
    yield "\n\n"
    yield from stream_report(FASTAPI_URL_PRELIM_REPORT, payload)
    yield "\n\n"
    yield from stream_report(FASTAPI_URL_BESTPRAC_REPORT, payload)


# ---- Reasoning ("thinking") panel -----------------------------------------
def stream_node_events(url: str, payload: dict):
    """Yield {label, detail} progress events from a graph-run SSE endpoint."""
    with requests.post(url, json=payload, stream=True) as response:
        if response.status_code != 200:
            yield {"label": f"⚠️ Error {response.status_code}", "detail": response.text[:300]}
            return
        for raw in response.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data: "):
                continue
            payload_str = raw[len("data: "):]
            try:
                yield json.loads(payload_str)
            except Exception:
                yield {"label": payload_str}


def _write_steps(steps):
    """Render a list of {label, detail} steps inside the current container."""
    for ev in steps:
        st.markdown(f"**{ev.get('label', '')}**")
        if ev.get("detail"):
            st.markdown(ev["detail"])


def render_reasoning(url: str, payload: dict, title: str = "🧠 Reasoning…"):
    """Live reasoning panel: stream node events into a collapsed st.status.

    Returns the collected steps so they can be persisted in chat history.
    """
    steps = []
    with st.status(title, expanded=False) as status:
        for ev in stream_node_events(url, payload):
            steps.append(ev)
            st.markdown(f"**{ev.get('label', '')}**")
            if ev.get("detail"):
                st.markdown(ev["detail"])
        status.update(label="✅ Reasoning complete", state="complete", expanded=False)
    return steps


def render_saved_reasoning(steps):
    """Re-render a persisted reasoning panel (collapsed) on script reruns."""
    with st.status("✅ Reasoning", state="complete", expanded=False):
        _write_steps(steps)


def render_history(history):
    """Render chat history, including persisted reasoning panels."""
    for msg in history:
        if msg.get("type") == "reasoning":
            with st.chat_message("assistant"):
                render_saved_reasoning(msg["steps"])
        else:
            st.chat_message(msg["role"]).markdown(msg["content"])

# Main Chat Interface
def main_chat():
    st.title("Mini-CDSS Chatbot")
    
    # Initialize conversation history and state variables
    if "main_chat_history" not in st.session_state:
        st.session_state.main_chat_history = []
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    if "main_graph_run_stat" not in st.session_state:
        st.session_state.main_graph_run_stat = None

    # Set initial assistant message if history is empty
    if not st.session_state.main_chat_history:
        initial_msg = (
            "👋 **Welcome to Mini-CDSS!** I’m your friendly **Medi-Doc 🩺 Assistant**, here to help with medical insights and patient reports.\n\n"
            "📂 **Have medical documents to upload?**\n\n"
            "No worries! Just **enable the RAG feature**, upload your files, and hit **Extract Medical Insights**.\n\n"
            "I’ll analyze them and pull out the key medical details for you! 🧑‍⚕️📄\n\n"
            "📝 **No documents? No problem!**\n"
            "Simply enter the **patient’s report** below. To ensure accuracy, please include the **age** and **sex** of the patient.\n\n"
            "💡 **Example Patient Report:**\n\n"
            "🧑‍⚕️ **42-year-old male** with a history of **type 2 diabetes** (diagnosed 10 years ago).\n"
            "🔍 **Current Complaints:** Polyuria, polydipsia, blurred vision, and fatigue for the past 2 weeks.\n"
            "⚕️ **Physical Exam:** Blood pressure **140/90 mmHg**, fundoscopic exam reveals **microaneurysms & cotton wool spots**.\n"
            "📊 **Investigations:** Random blood sugar **350 mg/dl**, HbA1c **10.5%**.\n"
            "💊 **Treatment Plan:** Intravenous fluids, insulin infusion, and electrolyte replacement initiated.\n\n"
            " If you have any medical documents upload them and click `Extract Medical Insights`, then enter `START` to begin and finally enter your patient age, sex, complaints below. \n\n Let’s get started!"
        
        )
        st.session_state.main_chat_history.append({"role": "assistant", "content": initial_msg})
    
    # Display chat history (including any persisted reasoning panels)
    render_history(st.session_state.main_chat_history)

    # Get new user input
    user_input = st.chat_input("Your patient report here...")
    if user_input:
        
        st.session_state.main_chat_history.append({"role": "user", "content": user_input})
        st.chat_message("user").markdown(user_input)

        # Step 1: Start the Process
        if user_input.strip().upper() == "START":
            if st.session_state.api_keys_validated == "validated":
                validation_data = {
                "tavily": st.session_state.tavily_api_key,
                "gemini": st.session_state.gemini_api_key,
                "groq": st.session_state.groq_api_key
                }

                # Send the keys to the validation endpoint
                try:
                    response = requests.post(FASTAPI_URL_SETTING_API, json=validation_data)
                    # print(response)
                    if response.status_code != 200:
                        st.error(f"Error {response.status_code}: {response.text}")
                except Exception as e:
                    st.error(f"Error during validation: {e}")


            response_msg = "Graph initiation started. Please enter your patient report."
            st.session_state.main_chat_history.append({"role": "assistant", "content": response_msg})
            st.chat_message("assistant").markdown(response_msg)

            # Set state to wait for report input
            st.session_state.main_graph_run_stat = "trigger_graph"

        # Step 2: Enter Report & Start Graph Processing
        elif st.session_state.get("main_graph_run_stat") == "trigger_graph":
            # Capture report input
            report_text = user_input.strip()
            st.session_state.main_graph_run_stat = "feedback"
            # print(st.session_state.medical_insights)
            payload = {
                "thread_id": st.session_state.thread_id,
                "text": report_text,
                "diagnosis_count": "3",
                "medical_report":st.session_state.get("medical_insights",""),
                "web_search_pro": st.session_state.get("web_search_pro", False)
            }

            # Surface the graph flow live in a collapsible reasoning panel
            with st.chat_message("assistant"):
                steps = render_reasoning(FASTAPI_URL_GRAPHSTART, payload, "🧠 Analyzing patient…")
            st.session_state.main_chat_history.append({"role": "assistant", "type": "reasoning", "steps": steps})

            # Stream the preliminary report live as it is generated
            with st.chat_message("assistant"):
                prelim_msg = st.write_stream(
                    stream_report(FASTAPI_URL_PRELIM_REPORT, {"thread_id": st.session_state.thread_id})
                )
            st.session_state.main_chat_history.append({"role": "assistant", "content": prelim_msg})

            # Add follow-up message from assistant
            follow_up_msg = "Kindly enter any feedback if you got any for us to retrigger the whole process, if not enter 'satisfied' to proceed!"
            st.session_state.main_chat_history.append({"role": "assistant", "content": follow_up_msg})
            st.chat_message("assistant").markdown(follow_up_msg)

        # Step 3: Feedback Loop
        elif st.session_state.get("main_graph_run_stat") == "feedback":
            user_feedback = user_input.strip()

            if not user_feedback:
                st.warning("⚠️ Feedback cannot be empty. Please provide feedback or type 'satisfied' to finalize.")
            
            # Step 4: Ending Loop
            elif user_feedback.lower() == "satisfied":
                # Finalize the process
                payload = {"thread_id": st.session_state.thread_id, "human_feedback": ""}
                st.session_state.main_graph_run_stat = "ending"
                st.session_state.medical_insights=""

                # Surface the finalize flow (disease extraction + web search) live
                with st.chat_message("assistant"):
                    steps = render_reasoning(FASTAPI_URL_PRELIM_INTERRUPT, payload, "🧠 Finalizing — searching best practices…")
                st.session_state.main_chat_history.append({"role": "assistant", "type": "reasoning", "steps": steps})

                # Stream the full report (NER + prelim + best practices) live
                with st.chat_message("assistant"):
                    final_report = st.write_stream(stream_final_reports(st.session_state.thread_id))

                st.session_state.main_chat_history.append({"role": "assistant", "content": final_report})

                # The backend captured each section while streaming; fetch the
                # styled PDF it cooked and keep it for the download button.
                st.session_state.report_pdf = fetch_report_pdf(st.session_state.thread_id)

                # Add follow-up message from assistant
                follow_up_msg = "A comprehensive patient report has been generated! You may clear the medical insights by clicking `Clear Previous Medical Insights` and enter 'START' to retrigger the graph for another patient!"
                st.session_state.main_chat_history.append({"role": "assistant", "content": follow_up_msg})
                st.chat_message("assistant").markdown(follow_up_msg)
            else:
                # Process additional feedback
                payload = {"thread_id": st.session_state.thread_id, "human_feedback": user_feedback}

                # Surface the re-diagnosis flow live
                with st.chat_message("assistant"):
                    steps = render_reasoning(FASTAPI_URL_PRELIM_INTERRUPT, payload, "🧠 Rethinking with your feedback…")
                st.session_state.main_chat_history.append({"role": "assistant", "type": "reasoning", "steps": steps})

                # Stream the updated preliminary report live
                with st.chat_message("assistant"):
                    st.markdown("🔄 **Updated Preliminary Report:**")
                    updated_prelim = st.write_stream(
                        stream_report(FASTAPI_URL_PRELIM_REPORT, {"thread_id": st.session_state.thread_id})
                    )
                updated_msg = f"🔄 **Updated Preliminary Report:**\n\n{updated_prelim}"
                st.session_state.main_chat_history.append({"role": "assistant", "content": updated_msg})

                # Add follow-up message from assistant
                follow_up_msg = "Kindly enter any feedback if you got any for us to retrigger the whole process, if not enter 'satisfied' to proceed!"
                st.session_state.main_chat_history.append({"role": "assistant", "content": follow_up_msg})
                st.chat_message("assistant").markdown(follow_up_msg)

    # Persisted download button for the finalized report PDF (survives reruns).
    if st.session_state.get("report_pdf"):
        st.download_button(
            "⬇️ Download report (PDF)",
            data=st.session_state.report_pdf,
            file_name=f"mini-cdss-report-{st.session_state.thread_id[:8]}.pdf",
            mime="application/pdf",
        )


def rag_chat():
    st.title("RAG Chat with Uploaded Medical Documents: ")

    # Initialize conversation history for RAG Chat if it doesn't exist.
    if "rag_chat_history" not in st.session_state:
        st.session_state.rag_chat_history = []

    # If no messages exist, add the initial message from the bot.
    if not st.session_state.rag_chat_history:
        initial_msg = "I am the RAG bot, kindly upload your documents and click on `Create Vector DB` to chat with me about the documents!"
        st.session_state.rag_chat_history.append({"role": "assistant", "content": initial_msg})



    # Display the conversation history (including persisted reasoning panels).
    render_history(st.session_state.rag_chat_history)

    # Get new input from the user.
    user_input = st.chat_input("Your message here...")

    if user_input:
        st.session_state.rag_chat_history.append({"role": "user", "content": user_input})
        st.chat_message("user").markdown(user_input)

        # Step 1: surface the retrieval flow live, then stream the answer.
        with st.chat_message("assistant"):
            steps = render_reasoning(
                FASTAPI_URL_RAG_SEARCH,
                {"thread_id": st.session_state.thread_id, "question": user_input},
                "🧠 Searching your documents…",
            )
        st.session_state.rag_chat_history.append({"role": "assistant", "type": "reasoning", "steps": steps})

        with st.chat_message("assistant"):
            answer = st.write_stream(
                stream_report(FASTAPI_URL_RAG_ANSWER, {"thread_id": st.session_state.thread_id})
            )
        st.session_state.rag_chat_history.append({"role": "assistant", "content": answer})





def main():
    st.set_page_config(page_title="Mini CDSS", layout="wide")

    # Create a session UUID
    if "thread_id" not in st.session_state:
        st.session_state.thread_id=str(uuid.uuid4())
    if "api_keys_validated" not in st.session_state:
        st.session_state.api_keys_validated=""

    if "gemini_api_key" not in st.session_state:
        st.session_state.gemini_api_key = ""
    if "groq_api_key" not in st.session_state:
        st.session_state.groq_api_key = ""
    if "tavily_api_key" not in st.session_state:
        st.session_state.tavily_api_key = ""
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = None
    if "medical_insight" not in st.session_state:
        st.session_state.medical_insight= ""

    if "enable_rag" not in st.session_state:
        st.session_state.enable_rag = False

    # Sidebar: shared across both chat interfaces.
    with st.sidebar:
        st.title("Settings")
        
        st.subheader("API Keys")
        st.caption("Not required — the backend runs on a local LLM and local embeddings.")
        st.text_input("Gemini API Key (unused)", key="gemini_api_key", type="password", disabled=True)
        st.text_input("GROQ API Key (unused)", key="groq_api_key", type="password", disabled=True)
        st.text_input("Tavily API Key (optional, set in backend .env)", key="tavily_api_key", type="password", disabled=True)

        # Connection check on Submit (just verifies the local backend/LLM is up).
        if st.button("Check Backend", type="primary"):
            try:
                response = requests.post(FASTAPI_URL_VALIDATE_API, json={})
                if response.status_code == 200:
                    st.success("Backend reachable and local LLM is up!")
                    st.session_state.api_keys_validated="validated"
                else:
                    st.error(f"Error {response.status_code}: {response.text}")
            except Exception as e:
                st.error(f"Error connecting to backend: {e}")

        # Search Pro: off -> SearXNG (local, free); on -> Tavily (premium web).
        st.session_state.web_search_pro = st.toggle(
            "Enable Search Pro",
            value=st.session_state.get("web_search_pro", False),
            help="Off: SearXNG (local). On: Tavily premium web search for the Best-Practices step.",
        )

        # Enable RAG toggle
        st.session_state.enable_rag = st.toggle("Enable RAG Chat (Upload Documents)", value=st.session_state.get("enable_rag", False))
        
        if st.session_state.enable_rag:
            # File uploader for documents
            uploaded_files = st.file_uploader("Upload Documents", type=["pdf"], accept_multiple_files=True)
            st.session_state.uploaded_files = uploaded_files if uploaded_files else []

            if st.button("Create Vector DB", type="primary"):
                if not st.session_state.uploaded_files:
                    st.error("Please upload at least one PDF file.")
                # elif not st.session_state.gemini_api_key:
                #     st.error("Please provide a Gemini API Key.")
                else:
                    # Prepare files for the POST request
                    files = [("files", (file.name, file.getvalue(), "application/pdf")) for file in st.session_state.uploaded_files]
                    data = {
                        "thread_id": st.session_state.thread_id,
                        }
                    
                    # Send request to FastAPI
                    with st.spinner("Creating Vector DB..."):
                        try:
                            response = requests.post(FASTAPI_URL_CREATE_VECTOR_DB, files=files, data=data)
                            if response.status_code == 200:
                                st.session_state.vector_db_checkbox = True
                                st.success("Vector DB created successfully! The DB will be maintained for 15 minutes or until the application is closed!")
                            else:
                                st.error(f"Error {response.status_code}: {response.text}")
                        except Exception as e:
                            st.error(f"An error occurred: {e}")
                    
            
            if st.button("Extract Medical Insights", type="primary"):
                st.session_state.medical_insight_checkbox = True

                # Ensure uploaded files exist before sending to FastAPI
                if not st.session_state.uploaded_files:
                    st.error("Please upload documents first.")
                else:
                    try:
                        # Prepare files for the POST request
                        files = [("files", (file.name, file.getvalue(), "application/pdf")) for file in st.session_state.uploaded_files]
                        data = {"thread_id": st.session_state.thread_id}  # Sending thread ID as form data

                        # Send request to FastAPI for medical insights extraction
                        with st.spinner("Extracting medical insights..."):
                            response = requests.post(FASTAPI_URL_EXTRACT_MEDICAL_TRIGGER, files=files, data=data)

                        if response.status_code == 200:
                            response = requests.post(FASTAPI_URL_EXTRACT_MEDICAL_DETAILS, json=data)
                            if response.status_code==200:
                                res = response.text.strip()
                                # Display the medical insights message
                                st.session_state.medical_insights = res
                                st.session_state.main_chat_history.append({"role": "assistant", "content": res})
                                st.toast("Medical Insights extracted!", icon="🩺")

                                follow_up_msg = "You can now enter `START` and then enter additional details for a comprehensive report!"
                                st.session_state.main_chat_history.append({"role": "assistant", "content": follow_up_msg})
                                st.chat_message("assistant").markdown(follow_up_msg)
                            else:
                                st.error(f"Error {response.status_code}: {response.text}")
                        else:
                            st.error(f"Error {response.status_code}: {response.text}")
                    except Exception as e:
                        st.error(f"An error occurred: {e}")

            if st.button("Clear Previous Medical Insights", type="primary"):
                st.session_state.medical_insights = ""
                st.session_state.medical_insight_checkbox = False
                st.success("Flushed Previous Medical Insights!")

                
            st.checkbox("Vector DB Created ✅", value=st.session_state.get("vector_db_checkbox", False), disabled=True)
            st.checkbox("Medical Insights Extracted ✅", value=st.session_state.get("medical_insight_checkbox", False), disabled=True)
                
        # Radio button to switch between chat interfaces.
        chat_option = st.radio("Select Chatbot", ("Main Chat", "RAG Chat") if st.session_state.enable_rag else ("Main Chat",))

        

    # Render the appropriate chat interface based on user selection.
    if chat_option == "Main Chat":
        main_chat()
    elif chat_option == "RAG Chat":
        rag_chat()


main()

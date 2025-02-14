import streamlit as st
# from validate_api import validate_tavily,validate_groq,validate_gemini
# from vectordb import create_vector_db
# from rag import rag_graph
# from main_graph import report_graph, get_graph
# from extract_medical_details import trigger_medical_insights_extraction
import os
import time
import uuid
import requests
import re

thread = {"configurable": {"thread_id":"1"}}
persist_directory =".chroma"

FASTAPI_URL_VALIDATE_API = "http://localhost:8000/validate_and_set_api"
FASTAPI_URL_SETTING_API = "http://localhost:8000/set_api"
FASTAPI_URL_GRAPHSTART = "http://localhost:8000/graphstart/"
FASTAPI_URL_PRELIM_INTERRUPT = "http://localhost:8000/prelimInterruptTrigger"
FASTAPI_URL_NER_REPORT = "http://localhost:8000/nerReport"
FASTAPI_URL_PRELIM_REPORT = "http://localhost:8000/prelimReport"
FASTAPI_URL_BESTPRAC_REPORT = "http://localhost:8000/bestpracReport"
FASTAPI_URL_CREATE_VECTOR_DB = "http://localhost:8000/addFilesAndCreateVectorDB"
FASTAPI_URL_RAG_SEARCH = "http://localhost:8000/ragSearch"
FASTAPI_URL_RAG_ANSWER = "http://localhost:8000/ragAnswer"
FASTAPI_URL_EXTRACT_MEDICAL_TRIGGER = "http://localhost:8000/extractMedicalDetails"
FASTAPI_URL_EXTRACT_MEDICAL_DETAILS = "http://localhost:8000/medicalInsightReport"

# Helper functions to get reports
def get_prelim_report(thread_id: str) -> str:
    payload = {"thread_id": thread_id}
    response = requests.post(FASTAPI_URL_PRELIM_REPORT, json=payload)
    if response.status_code == 200:
        # Expecting JSON like {"prelim_report": "..."}
        return response.text.strip()
    else:
        return f"Error: {response.text}"

def get_final_reports(thread_id: str) -> str:
    payload = {"thread_id": thread_id}
    ner_resp = requests.post(FASTAPI_URL_NER_REPORT, json=payload)
    prelim_resp = requests.post(FASTAPI_URL_PRELIM_REPORT, json=payload)
    bestprac_resp = requests.post(FASTAPI_URL_BESTPRAC_REPORT, json=payload)
    ner_report = ner_resp.text
    prelim_report = prelim_resp.text
    bestprac_report = bestprac_resp.text
    final_report = f"{ner_report}\n\n{prelim_report}\n\n{bestprac_report}"
    return final_report

# Main Chat Interface
def main_chat():
    st.title("Main Chatbot Interface")
    
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
            "🩺 **Let’s get started!** Enter your query below."
        )
        st.session_state.main_chat_history.append({"role": "assistant", "content": initial_msg})
    
    # Display chat history
    for msg in st.session_state.main_chat_history:
        st.chat_message(msg["role"]).markdown(msg["content"])
    

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
                "medical_report":st.session_state.get("medical_insights","")
            }

            with st.spinner("Processing graph..."):
                try:
                    response = requests.post(FASTAPI_URL_GRAPHSTART, json=payload)
                    if response.status_code == 200:
                        st.success("✅ Initial Prelim Report Generated!")
                    else:
                        st.error(f"❌ Graph execution failed: {response.text}")
                except Exception as e:
                    st.error(f"❌ Graph request error: {str(e)}")

            # Fetch preliminary report
            prelim = get_prelim_report(st.session_state.thread_id)
            prelim_msg = f"{prelim}"
            st.session_state.main_chat_history.append({"role": "assistant", "content": prelim_msg})
            st.chat_message("assistant").markdown(prelim_msg)

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

                with st.spinner("Finalizing report..."):
                    _ = requests.post(FASTAPI_URL_PRELIM_INTERRUPT, json=payload)

                    # Fetch final report
                    final_report = get_final_reports(st.session_state.thread_id)

                st.session_state.main_chat_history.append({"role": "assistant", "content": final_report})
                st.chat_message("assistant").markdown(final_report)

            else:
                # Process additional feedback
                payload = {"thread_id": st.session_state.thread_id, "human_feedback": user_feedback}

                with st.spinner("Updating preliminary report based on feedback..."):
                    _ = requests.post(FASTAPI_URL_PRELIM_INTERRUPT, json=payload)

                    # Fetch updated preliminary report
                    updated_prelim = get_prelim_report(st.session_state.thread_id)

                updated_msg = f"🔄 **Updated Preliminary Report:**\n\n{updated_prelim}"
                st.session_state.main_chat_history.append({"role": "assistant", "content": updated_msg})
                st.chat_message("assistant").markdown(updated_msg)


def process_rag(user_input):
    payload = {
        "thread_id": st.session_state.thread_id,
        "question": user_input,
        "gemini": st.session_state.gemini_api_key
    }
    
    try:
        # Step 1: Trigger RAG Search
        response = requests.post(FASTAPI_URL_RAG_SEARCH, json=payload)
        
        if response.status_code == 200:
            st.success("✅ RAG Search carried out, drafting answer...")
            
            # Step 2: Request the answer
            answer_payload = {"thread_id": st.session_state.thread_id}
            
            response = requests.post(FASTAPI_URL_RAG_ANSWER, json=answer_payload)
            
            if response.status_code == 200:
                final_answer = response.text.strip()
                return final_answer
            else:
                st.error(f"❌ Failed to fetch RAG answer: {response.text}")
        
        else:
            st.error(f"❌ RAG Search failed: {response.text}")

    except Exception as e:
        st.error(f"❌ Graph request error: {str(e)}")




def rag_chat():
    st.title("RAG Chat with Uploaded Medical Documents: ")
    
    # Initialize conversation history for RAG Chat if it doesn't exist.
    if "rag_chat_history" not in st.session_state:
        st.session_state.rag_chat_history = []

    # If no messages exist, add the initial message from the bot.
    if not st.session_state.rag_chat_history:
        initial_msg = "I am the RAG bot, for queries on the uploaded documents ask me here."
        st.session_state.rag_chat_history.append({"role": "assistant", "content": initial_msg})
    
    

    # Display the conversation history.
    for msg in st.session_state.rag_chat_history:
        st.chat_message(msg["role"]).markdown(msg["content"])

    # Get new input from the user.
    user_input = st.chat_input("Your message here...")

    if user_input:
        st.session_state.rag_chat_history.append({"role": "user", "content": user_input})
        st.chat_message("user").markdown(user_input)

        # Process the graph using the status indicator.
        answer = process_rag(user_input)
        # print(answer)
        st.chat_message("assistant").markdown(answer)
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
        gemini_api_key = st.text_input("Enter Gemini API Key", key="gemini_api_key", type="password")
        groq_api_key = st.text_input("Enter GROQ API Key", key="groq_api_key", type="password")
        tavily_api_key = st.text_input("Enter Tavily API Key", key="tavily_api_key", type="password")

        # Store the keys in session state once they are entered.
        if gemini_api_key != st.session_state.gemini_api_key:
            st.session_state.gemini_api_key = gemini_api_key
        if groq_api_key != st.session_state.groq_api_key:
            st.session_state.groq_api_key = groq_api_key
        if tavily_api_key != st.session_state.tavily_api_key:
            st.session_state.tavily_api_key = tavily_api_key
        
        # Validate keys on Submit button click
        if st.button("Submit", type="primary"):

            validation_data = {
                "tavily": st.session_state.tavily_api_key,
                "gemini": st.session_state.gemini_api_key,
                "groq": st.session_state.groq_api_key
            }

            # Send the keys to the validation endpoint
            try:
                response = requests.post(FASTAPI_URL_VALIDATE_API, json=validation_data)
                # print(response)
                if response.status_code == 200:
                    st.success("Keys validated and set in session env successfully!")
                    st.session_state.api_keys_validated="validated"
                else:
                    st.error(f"Error {response.status_code}: {response.text}")
            except Exception as e:
                st.error(f"Error during validation: {e}")

        # Enable RAG toggle
        st.session_state.enable_rag = st.toggle("Enable RAG Chat (Upload Documents)", value=st.session_state.get("enable_rag", False))
        
        if st.session_state.enable_rag:
            # File uploader for documents
            uploaded_files = st.file_uploader("Upload Documents", type=["pdf"], accept_multiple_files=True)
            st.session_state.uploaded_files = uploaded_files if uploaded_files else []

            if st.button("Create Vector DB", type="primary"):
                if not st.session_state.uploaded_files:
                    st.error("Please upload at least one PDF file.")
                elif not st.session_state.gemini_api_key:
                    st.error("Please provide a Gemini API Key.")
                else:
                    # Prepare files for the POST request
                    files = [("files", (file.name, file.getvalue(), "application/pdf")) for file in st.session_state.uploaded_files]
                    data = {
                        "thread_id": st.session_state.thread_id,
                        "gemini_api_key": st.session_state.gemini_api_key,
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

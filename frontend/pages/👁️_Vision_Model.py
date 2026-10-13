import streamlit as st
import requests
import uuid
import os
from dotenv import load_dotenv
from PIL import Image
import io

load_dotenv()

# Backend origin is env-driven (see .env) — switch local/hosted in one place.
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")

# Define API endpoints
INPUT_IMAGE_URL = f"{BACKEND_URL}/input-image/"
INPUT_QUERY_URL = f"{BACKEND_URL}/input-query/"
VISION_ANSWER_URL = f"{BACKEND_URL}/vision-answer/"
VISION_FEEDBACK_URL = f"{BACKEND_URL}/vision-feedback/"

# Initialize session state variables
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "conversation_stage" not in st.session_state:
    st.session_state.conversation_stage = "upload"
if "coversation_substage" not in st.session_state:
    st.session_state.coversation_substage = "query"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None

def display_chat():
    """Display the chat history with image support."""
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            if msg["type"] == "image":
                st.image(msg["content"], caption="Uploaded Image", width=250)
            else:
                st.markdown(msg["content"])

st.title("Vision Chatbot")

# ---- Stage 1: Upload Image ----
if st.session_state.conversation_stage == "upload":
    st.write("**Step 1:** Upload an image to start.")
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        # Read image
        image = Image.open(uploaded_file)
        st.session_state.uploaded_image = image  # Store image in session state

        # Convert image to bytes for API request
        img_bytes = io.BytesIO()
        image.save(img_bytes, format="PNG")
        img_bytes = img_bytes.getvalue()

        # Send image to API
        files = {"image": (uploaded_file.name, img_bytes, uploaded_file.type)}
        data = {"thread_id": st.session_state.thread_id}
        with st.spinner("Uploading image..."):
            response = requests.post(INPUT_IMAGE_URL, data=data, files=files)

        if response.status_code == 200:
            # Add image to chat history
            st.session_state.chat_history.append({
                "role": "assistant",
                "type": "image",
                "content": st.session_state.uploaded_image
            })
            st.session_state.chat_history.append({
                "role": "assistant",
                "type": "text",
                "content": "Image uploaded successfully. Enter your query now."
            })

            st.session_state.conversation_stage = "chatting"
            st.rerun()

# ---- Stage 2: Query Chat ----
elif st.session_state.conversation_stage == "chatting":
    display_chat()
    
    user_query = st.chat_input("Enter your query")
    if user_query:

        # ---- Stage 3: Reset Chat if needed ----
        if user_query.lower().strip() == "reset":
            # Reset session state
            st.session_state.chat_history = []  # Clear chat history
            st.session_state.conversation_stage = "upload"  # Reset stage
            st.session_state.coversation_substage=="query"
            st.session_state.uploaded_image = None  # Remove image
            st.rerun()  # Refresh UI

        if st.session_state.coversation_substage=="query":
            st.session_state.chat_history.append({"role": "user", "type": "text", "content": user_query})

            # Send query to API
            data = {"thread_id": st.session_state.thread_id, "query": user_query}
            with st.spinner("Processing query..."):
                response = requests.post(INPUT_QUERY_URL, json=data)

            if response.status_code == 200:
                # Fetch answer from API
                answer_response = requests.post(VISION_ANSWER_URL, json={"thread_id": st.session_state.thread_id})

                if answer_response.status_code == 200:
                    answer = answer_response.text
                    st.session_state.chat_history.append({"role": "assistant", "type": "text", "content": answer})
                    st.session_state.coversation_substage="feedback"
                else:
                    st.session_state.chat_history.append({"role": "assistant", "type": "text", "content": "Failed to retrieve answer."})
            else:
                st.session_state.chat_history.append({"role": "assistant", "type": "text", "content": "Error processing query."})

            st.session_state.chat_history.append({"role": "assistant", "type": "text", "content": "Enter any further feedback (if any) or enter `reset` to reset the chat."})
            st.rerun()
            
        
        if st.session_state.coversation_substage=="feedback":
            st.session_state.chat_history.append({"role": "user", "type": "text", "content": user_query})

            # Send query to API
            data = {"thread_id": st.session_state.thread_id, "feedback": user_query}
            with st.spinner("Processing query..."):
                response = requests.post(VISION_FEEDBACK_URL, json=data)

            if response.status_code == 200:
                # Fetch answer from API
                answer_response = requests.post(VISION_ANSWER_URL, json={"thread_id": st.session_state.thread_id})

                if answer_response.status_code == 200:
                    answer = answer_response.text
                    st.session_state.chat_history.append({"role": "assistant", "type": "text", "content": answer})
                else:
                    st.session_state.chat_history.append({"role": "assistant", "type": "text", "content": "Failed to retrieve answer."})
            else:
                st.session_state.chat_history.append({"role": "assistant", "type": "text", "content": "Error processing query."})

            st.session_state.chat_history.append({"role": "assistant", "type": "text", "content": "Enter any further feedback (if any) or enter `reset` to reset the chat."})
            st.rerun()




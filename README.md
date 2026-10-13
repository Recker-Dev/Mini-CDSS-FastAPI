# FastAPI Backend Logic

## 📌 Project Overview

FASTAPI Backend Server Code to efficiently handle code for Mini-CDSS Project.

## 📖 Description

Mini-CDSS helps doctors to take an initial patient encounter and creates an initial patient report, give diagnosis and even use Tavily Web Search to find all the relevant Best Practises.
The Doctor can also extract medical insights from any uploaded documents before processing with the reporting system. We also allow the doctor to RAG - Chat with the uploaded documents.
The Doctor can also use our vision model to upload any images and query from it, give feedback etc.

AIM: Founding Steps of Next-gen CDSS systems, incorporating Gen-AI in report building, multimodal capabilities, CDSS is a highly rigid rule based system, Gen-AI introduces flexibility, context-awareness for fundamental systems to work on, and leveraging frameworks like Langgraph and Pydantic it is possible not only to reduce LLM Hallucination but also introduce dynamic real-time decision branches based on simple text.

## 🏗️ Tech Stack

- **Language:** Python 3.11+
- **Framework:** FastAPI
- **GenAI Framework:** Langgraph
- **LLM:** A **local** OpenAI-compatible model (default: Qwen3.5 via llama.cpp), accessed with the official `openai` client. No hosted LLM provider is used.
- **Embeddings:** Local `all-MiniLM-L6-v2` (ChromaDB default, runs on CPU, no API key).
- **Web Search Tool:** Tavily (optional — skipped gracefully if no key).
- **Database (Vector-DB):** ChromaDB [linked with session uuid for 15 minutes before the db is flushed].
- **Package manager:** [uv](https://docs.astral.sh/uv/)

> **Note on vision:** The original project used Gemini/Groq vision models. The default local model is text-only, so the `/input-image/`, `/input-query/`, `/vision-answer/`, and `/vision-feedback/` endpoints return a clear "not supported" response. To re-enable, serve a multimodal model and restore `config/vision_graph.py`.

## 📂 Project Structure

```
.
├── config
│   ├── __init__                 # Init Package
│   ├── llm                      # Shared local LLM (OpenAI client) + local embedding function.
│   ├── prompts.py               # Loads system prompts from prompts/*.md (cached).
│   ├── progress.py              # Maps graph node updates -> frontend progress events.
│   ├── pdf.py                   # Renders the report markdown into a styled (warm/minimal) PDF.
│   ├── report_store.py          # Captures streamed report text per thread (source for the PDF).
│   ├── fastapi_models           # API Endpoint pydantic structure
│   ├── hugging_face_ner         # Runs Hugging Face model on intial patient encounter data to extract relevant entities, and returns an entity dict.
│   ├── main_graph               # The main code dedicated to NER Report, Prelim Report, Feedback Loop and Best Practise Report Generation Logic.
│   ├── medical_summarizer_graph # Extracts the relevant medical details from uploaded documents.
│   ├── rag.py                   # A REACT Blueprint of a smarter RAG Agent.
│   ├── validate_api.py          # Health check for the local LLM.
│   ├── vectordb.py              # Creation/deletion logic of Vector-DBs (local embeddings).
│   ├── vision_graph.py          # Vision disabled stub (text-only local model).
│
├── cron                         # Server side Background tasks.
│   ├── __init__                 # Init Package
│   ├── storage                  # Keeps track of vector_db_list, tracking current Vec DBs on the server and their lifetime.
│   ├── jobs                     # Secheduled Jobs that triggers at a given moment/interval.
│   ├── tasks                    # Logic of the Jobs scheduled.
│
│
├── prompts/                     # System prompts as editable .md files (one per prompt).
├── frontend/                    # Streamlit UI (1_🩺_Homepage.py, pages/).
├── Dockerfile                   # Single image for both services.
├── docker-compose.yml           # minicdss-backend + minicdss-frontend (infra from .env).
├── .env / .env.example          # Env vars (.env gitignored; .env.example committed).
├── main                         # Main Server Side logic with endpoints.
├── pyproject.toml               # Project + dependency definitions (uv).
├── uv.lock                      # Locked, reproducible dependency versions.
├── requirements.txt             # Auto-exported from uv.lock (for pip/render fallback).
├── render.yaml                  # Config file to run Server on render.
```

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Recker-Dev/Mini-CDSS-FastAPI.git
cd Mini-CDSS-FastAPI
```

### 2️⃣ Install Dependencies (uv)

This is a single repo containing **both** the FastAPI backend and the Streamlit
frontend (`frontend/`), split via dependency extras:

```bash
uv sync --all-extras          # everything (dev)
# or, for a lean install of just one side:
uv sync --extra backend       # API server only
uv sync --extra frontend      # Streamlit UI only
```

### 3️⃣ Run a local LLM

Any OpenAI-compatible server works (llama.cpp, Ollama, vLLM, …). Example sanity check:

```bash
curl http://localhost:55555/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.5","messages":[{"role":"user","content":"Hi"}],
       "chat_template_kwargs":{"enable_thinking":false},"stream":false}'
```

### 4️⃣ Configure Environment Variables

Create a `.env` file (see the committed template):

- `LOCAL_LLM_BASE_URL`  — e.g. `http://localhost:55555/v1` (required)
- `LOCAL_LLM_MODEL`     — e.g. `qwen3.5` (required)
- `LOCAL_LLM_API_KEY`   — usually ignored by local servers; defaults to `local`
- `TAVILY_API_KEY`      — optional, enables the Best-Practices web search
- `HF_TOKEN`            — optional, enables the HuggingFace Medical-NER augmentation

Embeddings are fully local — no key needed.

### 6️⃣ Start the backend and frontend

Two processes (one terminal each), both from the repo root so they share `.env`:

```bash
# Backend API — keep it on port 8000 to match the frontend's BACKEND_URL default
uv run uvicorn main:app --host 0.0.0.0 --port 8000

# Frontend UI
uv run streamlit run "frontend/1_🩺_Homepage.py"
```

The frontend reads `BACKEND_URL` from `.env` (one toggle to switch local/hosted).
Reports and the RAG answer stream in live, and the graph flow is surfaced in a
collapsible "🧠 Reasoning" panel (identified conditions, best-practice sources, …).

### 7️⃣ Access API Documentation

- Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- ReDoc: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

## 📜 API Endpoints

| Method | Endpoint                     | Description                                                          |
| ------ | ---------------------------- | -------------------------------------------------------------------- |
| POST   | `/validate_and_set_api`      | API configuration                                                    |
| POST   | `/set_api`                   | Helps to set the keys just before graph starts.                      |
| POST   | `/graphstart/`               | Used to start main graph                                             |
| POST   | `/prelimInterruptTrigger`    | Takes feedback from user based on prelim diagnosis report generated. |
| POST   | `/nerReport`                 | Returns the NER Report of patient encounter.                         |
| POST   | `/prelimReport`              | Returns the Prelim Report of patient encounter.                      |
| POST   | `/bestpracReport`            | Returns the Best Practises Report of patient encounter.              |
| GET    | `/reportPdf?thread_id=`      | Styled PDF of the finalized report (NER + prelim + best practices).  |
| GET    | `/health`                    | Aggregate status: backend, local LLM, Tavily.                        |
| POST   | `/addFilesAndCreateVectorDB` | Logic to create a vector DB and overwrite if existing.               |
| POST   | `/ragSearch`                 | Used to search the docs.                                             |
| POST   | `/ragAnswer`                 | Returns the rag answer.                                              |
| POST   | `/extractMedicalDetails`     | Used to get Medical Details from the uploaded documents.             |
| POST   | `/medicalInsightReport`      | Returns the Medical Insights gathered from the uploaded documents..  |
| POST   | `/input-image/`              | Used to input image to vision model.                                 |
| POST   | `/input-query/`              | Enter the query about the image.                                     |
| POST   | `/vision-answer/`            | Returns the answer generated.                                        |
| POST   | `/vision-feedback/`          | Takes feedback for vision output.                                    |

## 📝 License

This project is licensed under the MIT License.

## 📬 Contact

For any issues or suggestions, reach out at [reckerdev@gmail.com](mailto:reckerdev@gmail.com).

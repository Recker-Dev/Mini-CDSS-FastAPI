# FastAPI Backend Logic

## 📌 Project Overview
FASTAPI Backend Server Code to efficiently handle code for Mini-CDSS Project.


## 📖 Description
Mini-CDSS helps doctors to take an intiial patient encounter and creates an initial patient report, give diagnosis and even use Tavily Web Search to find all the relevant Best Practises.
The Doctor can also extract medical insights from any uploaded documents before processing with the reporting system. We also allow the doctor to RAG - Chat with the uploaded documents.
The Doctor can also use our vision model to upload any iamges and query from it, give feedback etc.

AIM: Founding Steps of Next-gen CDSS systems, incorporating Gen-AI in report building, multimodal capabilities, CDSS is a highly rigid rule based system, Gen-AI introduces flexibility, context-awareness for fundamental systems to work on, and leveraging frameworks like Langgraph and Pydantic it is possible not only to reduce LLM Hallucination but also introduce dynamic real-time decision branches based on simple text.


## 🏗️ Tech Stack
- **Language:** Python 3.x
- **Framework:** FastAPI
- **GenAI Framework:** Langgraph
- **Generative Services Used:** Gemini, Groq
- **Web Search Tool** Tavily
- **Database(Vector-DB):** ChromaDB [persisted on hosted server;linked with session uuid for a duration of 15 minutes before the db is flushed]


## 📂 Project Structure
```
.
├── config
│   ├── __init__                 # Init Package
│   ├── credentials              # Loads in base-64 encoded Credentails of Google Service Account (needed to run gemini on server.)
│   ├── fastapi_models           # API Endpoint pydantic structure
│   ├── hugging_face_ner         # Runs Hugging Face model on intial patient encounter data to extract relevant entities, and returns an entity dict.
│   ├── main_graph               # The main code dedicated to NER Report, Prelim Report, Feedback Loop and Best Practise Report Generation Logic.
│   ├── medical_summarizer_graph # Extracts the relevant medical details from uploaded documents.
│   ├── rag.py                   # A REACT Blueprint of a smarter RAG Agent.
│   ├── validate_api.py          # Function to validate the entered API Keys.
│   ├── vectordb.py              # Function dedicated to creation and deletion logic of Vector-DBs.
│   ├── vision_graph.py          # Brings Multi-modal capabilities to our project.
│
├── cron                         # Server side Background tasks.
│   ├── __init__                 # Init Package
│   ├── storage                  # Keeps track of vector_db_list, tracking current Vec DBs on the server and their lifetime.
│   ├── jobs                     # Secheduled Jobs that triggers at a given moment/interval.
│   ├── tasks                    # Logic of the Jobs scheduled.
│
│
├── .env                         # Tracks the env variables and their details
├── main                         # Main Server Side logic with endpoints.
├── requirements                 # Necessary dependencies to run the project   
├── render.yaml                  # COnfig file to run Server on render.
```


### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Recker-Dev/Mini-CDSS-FastAPI.git
cd Mini-CDSS-FastAPI
```

### 2️⃣ Set Up Virtual Environment
```bash
conda create --name test_env python=3.10 -y
conda activate test_env
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Configure Environment Variables
Create `.env` file and put "GOOGLE_API_KEY" ,"TAVILY_API_KEY","GROQ_API_KEY","HF_TOKEN","GOOGLE_CREDENTIALS_BASE64"


### 6️⃣ Start the FastAPI Server
```bash
uvicorn app.main:app --reload
```

### 7️⃣ Access API Documentation
- Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- ReDoc: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)



## 📜 API Endpoints
| Method | Endpoint                   | Description          |
|--------|----------------------------|----------------------|
| POST    | `/validate_and_set_api`    | API configuration    |
| POST   | `/set_api`                 | Helps to set the keys just before graph starts.                |
| POST    | `/graphstart/`             | Used to start main graph       |
| POST    | `/prelimInterruptTrigger`             | Takes feedback from user based on prelim diagnosis report generated.       |
| POST    | `/nerReport`             | Returns the NER Report of patient encounter.      |
| POST    | `/prelimReport`             | Returns the Prelim Report of patient encounter.      |
| POST    | `/bestpracReport`             | Returns the Best Practises Report of patient encounter.       |
| POST    | `/addFilesAndCreateVectorDB`             | Logic to create a vector DB and overwrite if existing.       |
| POST    | `/ragSearch`             | Used to search the docs.      |
| POST    | `/ragAnswer`             | Returns the rag answer.       |
| POST    | `/extractMedicalDetails`             | Used to get Medical Details from the uploaded documents.       |
| POST    | `/medicalInsightReport`             | Returns the Medical Insights gathered from the uploaded documents..       |
| POST    | `/input-image/`             | Used to input image to vision model.       |
| POST    | `/input-query/`             | Enter the query about the image.       |
| POST    | `/vision-answer/`             | Returns the answer generated.       |
| POST    | `/vision-feedback/`             |Takes feedback for vision output.  |






## 📝 License
This project is licensed under the MIT License.

## 📬 Contact
For any issues or suggestions, reach out at [reckerdev@gmail.com](mailto:reckerdev@gmail.com).


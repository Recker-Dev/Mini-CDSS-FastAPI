## FAST-API BASE APP
from config.vectordb import create_vector_db
from langchain_community.document_loaders import PyPDFLoader
from fastapi import FastAPI, HTTPException, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import datetime
import base64

from typing import List

from config.fastapi_models import Thread,GraphInput,PrelimInterrupt,APIInput,RagChat,VisionInput,VisionFeedback
from config.validate_api import validate_keys, health_report
from config.main_graph import graph, stream_ner_report, stream_prelim_report, stream_bestprac_report
from config.rag import rag_graph, stream_rag_answer
from config.medical_summarizer_graph import medical_insights_graph, stream_medical_report
from config.vision_graph import vision_graph
from config.progress import sse, main_graph_event, rag_event


from cron.jobs import scheduler
from cron.tasks import appendVectorName

app = FastAPI()

scheduler.start()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allows all headers
)

load_dotenv()  


@app.get("/ping")
def root():
    return "PONG"

@app.get("/health")
def health():
    """Aggregate health: backend, local LLM, and Tavily web search."""
    return health_report()

def my_shutdown_job():
    print(f"Server shutting down at {datetime.datetime.now()}")

@app.on_event("shutdown")
def on_shutdown():
    my_shutdown_job()
    scheduler.shutdown()  # Stop scheduler cleanly



@app.post("/validate_and_set_api")
async def validating_api(input_data: APIInput):
    # The only hard dependency now is the local LLM. Tavily (web search) is
    # optional; if a key is provided we store it, otherwise web search is skipped.
    response = validate_keys()
    if response == "Validated":
        if input_data.tavily:
            os.environ["TAVILY_API_KEY"] = input_data.tavily
    else:
        raise HTTPException(status_code=400, detail=response)

    return {"response":response}

@app.post("/set_api")
async def setting_api(input_data: APIInput):
    if input_data.tavily:
        os.environ["TAVILY_API_KEY"] = input_data.tavily


@app.post("/graphstart/")
async def graphstart(input_data: GraphInput):
    # logger.debug(f'{input_data}')
    async def event_stream():
        thread = {"configurable": {"thread_id": input_data.thread_id}}
        for event in graph.stream({"initial_summary": input_data.text,
                                   "diagnosis_count": input_data.diagnosis_count,
                                   "medical_report": input_data.medical_report}, thread, stream_mode="updates"):
            for node_name, update in event.items():
                ev = main_graph_event(node_name, update)
                if ev:
                    yield sse(ev)

    return StreamingResponse(event_stream(), media_type="text/event-stream")




@app.post("/prelimInterruptTrigger")
async def prelim_human_feedback(prelim_feedback: PrelimInterrupt):
    thread = {"configurable": {"thread_id": prelim_feedback.thread_id}}
    further_feedback = prelim_feedback.human_feedback
    graph.update_state(thread, {"human_prelim_feedback":further_feedback}, as_node="prelim human feedback node")

    async def event_stream():
        for event in graph.stream(None, thread, stream_mode="updates"):
            for node_name, update in event.items():
                ev = main_graph_event(node_name, update)
                if ev:
                    yield sse(ev)

    return StreamingResponse(event_stream(), media_type="text/event-stream")
    


@app.post("/nerReport")
async def ner_report(thread: Thread):
    thread = {"configurable": {"thread_id": thread.thread_id}}
    post_ner_data = graph.get_state(thread).values.get('post_ner_data')

    async def event_stream():
        if post_ner_data is None:
            yield "No NER data available. Run /graphstart/ first."
            return
        for chunk in stream_ner_report(post_ner_data):
            yield chunk

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.post("/prelimReport")
async def prelim_report(thread: Thread):
    thread = {"configurable": {"thread_id": thread.thread_id}}
    diagnoses = graph.get_state(thread).values.get('diagnoses')

    async def event_stream():
        if diagnoses is None:
            yield "No diagnoses available. Run /graphstart/ first."
            return
        for chunk in stream_prelim_report(diagnoses):
            yield chunk

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.post("/bestpracReport")
async def best_prac_report(thread: Thread):
    thread = {"configurable": {"thread_id": thread.thread_id}}
    best_practises = graph.get_state(thread).values.get('best_practises')

    async def event_stream():
        if not best_practises:
            yield "No best-practices data available. Complete the diagnosis flow first."
            return
        for chunk in stream_bestprac_report(best_practises):
            yield chunk

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.post("/addFilesAndCreateVectorDB")
async def add_files(
    thread_id: str= Form(...),
    files: List[UploadFile] = File(...),
):
    """
    Endpoint to upload PDF files and create a vector database.
    Embeddings are computed locally; no API key is required.
    """
    # DEBUG: Check if files are received correctly

    try:
        success, message = await create_vector_db(files, thread_id)
        if not success:
            raise HTTPException(status_code=400, detail=message)
        
        # scheduler.add_job(appendVectorName, args=[f"{thread_id}_vectorDB"])
        # Directly trigger the append function
        appendVectorName(f"{thread_id}_vectorDB")
        return {"success": success, "message": message}

    except HTTPException as http_err:
        raise http_err  # Re-raise the HTTPException directly

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
    
@app.post("/ragSearch")
async def rag_chat(input_data: RagChat):
    COLLECTION_NAME="vectorDB"

    async def event_stream():
        thread = {"configurable": {"thread_id": input_data.thread_id}}
        for event in rag_graph.stream({
            "question": input_data.question,
            "max_queries": 3,
            "collection_path": f"{input_data.thread_id}_{COLLECTION_NAME}",
            "allowed_call_count": 2,
            "expired_call_count": 0
        }, thread):
            for node_name, update in event.items():
                ev = rag_event(node_name, update)
                if ev:
                    yield sse(ev)

    return StreamingResponse(event_stream(), media_type="text/event-stream")
    

@app.post("/ragAnswer")
async def rag_answer(thread: Thread):
    thread = {"configurable": {"thread_id": thread.thread_id}}
    values = rag_graph.get_state(thread).values
    question = values.get('question')
    docs_retrieved = values.get('docs_retrieved')
    relevance = values.get('relevance')

    def event_stream():
        if question is None:
            yield "No query found. Run /ragSearch first."
            return
        for chunk in stream_rag_answer(question, docs_retrieved, relevance):
            yield chunk

    return StreamingResponse(event_stream(), media_type="text/event-stream")



@app.post("/extractMedicalDetails")
async def extract_medical_details(
    files: List[UploadFile] = File(...),
    thread_id: str = Form(...)
):
    # if files:
    #     return {"status" :"ok"}
    
    thread = {"configurable": {"thread_id":thread_id}}
    async def readFiles(files):
        extracted_files=[]
        for file_id,uploaded_file in enumerate(files):
            file_path =  f"temp_{uploaded_file.filename}" 
            with open(file_path, "wb") as f:
                f.write(await uploaded_file.read())
            # Load and split the PDF into pages.
            loader = PyPDFLoader(file_path)
            pages = loader.load_and_split()
            extracted_files.append((file_id,pages))
            os.remove(file_path)

        return extracted_files
    

    files = await readFiles(files)

    async def event_stream():
        try:
            for event in medical_insights_graph.stream({"files":files}, thread):
                node_name = next(iter(event.keys()))
                yield f"data: Processing node: {node_name}\n\n"

        except Exception as e:
            yield {"error": str(e)}

    return StreamingResponse(event_stream(), media_type="text/event-stream")






@app.post("/medicalInsightReport")
async def medical_report_insight(thread: Thread):
    thread = {"configurable": {"thread_id": thread.thread_id}}
    medical_insights = medical_insights_graph.get_state(thread).values.get('medical_insights')

    async def event_stream():
        if not medical_insights:
            yield "No medical insights available. Run /extractMedicalDetails first."
            return
        for chunk in stream_medical_report(medical_insights):
            yield chunk

    return StreamingResponse(event_stream(), media_type="text/event-stream")



# --- Vision endpoints -------------------------------------------------------
# The vision graph is intact but currently disabled (text-only local model);
# every call flows through the graph and returns a clear "not supported" answer.
# Re-enable by serving a multimodal model and flipping VISION_SUPPORTED in
# config/vision_graph.py.

@app.post("/input-image/")
async def input_image(thread_id:str=Form(...), image: UploadFile = File(...)):
    """API endpoint to start vision graph with an image."""

    async def start_graph(thread_id:str, base64_image: str):
        """Invoke graph with base64 image."""
        thread = {"configurable": {"thread_id": thread_id}}
        vision_graph.invoke({"base64_image": base64_image}, thread,)

    try:
        # Convert image to Base64
        image_bytes = await image.read()
        base64_image = base64.b64encode(image_bytes).decode("utf-8")

        # Trigger graph function and await result
        await start_graph(thread_id, base64_image)

        return {"graph started, image input success!"}

    except Exception as e:
        return {f"graph failed, error: {e}"}

@app.post("/input-query/")
async def input_query_for_image(input_data:VisionInput):

    async def resume_graph(thread_id:str, query: str):
        thread = {"configurable": {"thread_id": thread_id}}
        vision_graph.update_state(thread,{"query":query},as_node="enter query")
        vision_graph.invoke(None, thread)

    try:
        await resume_graph(input_data.thread_id,input_data.query)
        return {"Graph was resumed and query was taken into account. Check formed answer"}
    except Exception as e:
        return {f"graph failed, error: {e}"}


@app.post("/vision-answer/")
async def process_image_answer(input_data:Thread):


    thread = {"configurable": {"thread_id": input_data.thread_id}}
    async def event_stream():
            final_state = vision_graph.get_state(thread)
            answer = final_state.values.get('answer')
            yield f"{answer}"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/vision-feedback/")
async def process_vision_feedback(input_data:VisionFeedback):

    async def resume_graph(thread_id:str, feedback: str):
        thread = {"configurable": {"thread_id": thread_id}}
        vision_graph.update_state(thread, {"feedback":feedback}, as_node="human feedback")
        vision_graph.invoke(None,thread)

    try:
        await resume_graph(input_data.thread_id,input_data.feedback)
        return {"Graph was resumed and feedback was taken into account."}
    except Exception as e:
        return {f"graph failed, error: {e}"}



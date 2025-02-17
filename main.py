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

from typing import List, Optional

from config.fastapi_models import Thread,GraphInput,PrelimInterrupt,APIInput,RagChat
from config.validate_api import validate_keys
from config.main_graph import graph
from config.rag import rag_graph
from config.medical_summarizer_graph import medical_insights_graph
from config.vision_graph import vision_graph


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

def my_shutdown_job():
    print(f"Server shutting down at {datetime.datetime.now()}")

@app.on_event("shutdown")
def on_shutdown():
    my_shutdown_job()
    scheduler.shutdown()  # Stop scheduler cleanly



@app.post("/validate_and_set_api")
async def validating_api(input_data: APIInput):
    response=validate_keys(tav_api=input_data.tavily,gemini_api=input_data.gemini,groq_api=input_data.groq)
    if response =="Validated":
        os.environ["TAVILY_API_KEY"] = input_data.tavily
        os.environ["GEMINI_API_KEY"] = input_data.gemini
        os.environ["GROQ_API_KEY"] = input_data.groq
    else:
        raise HTTPException(status_code=400, detail=response)
    
    return {"response":response}

@app.post("/set_api")
async def setting_api(input_data: APIInput):
    os.environ["TAVILY_API_KEY"] = input_data.tavily
    os.environ["GEMINI_API_KEY"] = input_data.gemini
    os.environ["GROQ_API_KEY"] = input_data.groq


@app.post("/graphstart/")
async def graphstart(input_data: GraphInput):
    # logger.debug(f'{input_data}')
    async def event_stream():
        thread = {"configurable": {"thread_id": input_data.thread_id}}
        for event in graph.stream({"initial_summary": input_data.text, 
                                   "diagnosis_count": input_data.diagnosis_count,
                                   "medical_report": input_data.medical_report}, thread, stream_mode="updates"):
            node_name = next(iter(event.keys()))
            yield f"data: {node_name}\n\n"
        
    return StreamingResponse(event_stream(), media_type="text/event-stream")


            # await asyncio.sleep(1)




@app.post("/prelimInterruptTrigger")
async def prelim_human_feedback(prelim_feedback: PrelimInterrupt):
    thread = {"configurable": {"thread_id": prelim_feedback.thread_id}}
    further_feedback = prelim_feedback.human_feedback
    graph.update_state(thread, {"human_prelim_feedback":further_feedback}, as_node="prelim human feedback node")

    async def event_stream():
        for event in graph.stream(None, thread, stream_mode="updates"):
            node_name = next(iter(event.keys()))
            yield f"data: {node_name}\n\n"
            # await asyncio.sleep(1)
        
    return StreamingResponse(event_stream(), media_type="text/event-stream")
    


@app.post("/nerReport")
async def ner_report(thread: Thread):
    # final_state = graph.get_state(thread)
    # ner_report = final_state.values.get('ner_report')
    # return {"ner_report":ner_report}
    thread = {"configurable": {"thread_id": thread.thread_id}}
    async def event_stream():
        final_state = graph.get_state(thread)
        ner_report = final_state.values.get('ner_report')
        yield f"{ner_report}"
        
    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.post("/prelimReport")
async def prelim_report(thread: Thread):
    # thread = {"configurable": {"thread_id": thread.thread_id}}
    # final_state = graph.get_state(thread)
    # prelim_report = final_state.values.get('prelim_report')
    # return {"prelim_report":prelim_report}

    thread = {"configurable": {"thread_id": thread.thread_id}}
    async def event_stream():
        final_state = graph.get_state(thread)
        prelim_report = final_state.values.get('prelim_report')
        yield f"{prelim_report}"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.post("/bestpracReport")
async def best_prac_report(thread: Thread):
    thread = {"configurable": {"thread_id": thread.thread_id}}
    async def event_stream():
        final_state = graph.get_state(thread)
        best_prac_report = final_state.values.get('best_practise_report')
        yield f"{best_prac_report}"
        
    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.post("/addFilesAndCreateVectorDB")
async def add_files(
    thread_id: str= Form(...),
    gemini_api_key: Optional[str] = Form(None),
    files: List[UploadFile] = File(...),
):
    """
    Endpoint to upload PDF files and create a vector database.
    The API key for the embedding function is supplied as a form field.
    """
    # DEBUG: Check if files are received correctly

    try:
        if not gemini_api_key:
            gemini_api_key = os.environ["GOOGLE_API_KEY"] 
        success, message = await create_vector_db(files, gemini_api_key,thread_id)
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
    if not input_data.gemini:
        gemini = os.environ["GOOGLE_API_KEY"]
    else:
        gemini = input_data.gemini
        
    COLLECTION_NAME="vectorDB"

    async def event_stream():
        thread = {"configurable": {"thread_id": input_data.thread_id}}
        for event in rag_graph.stream({
            "question": input_data.question,
            "max_queries": 3,
            "collection_path": f"{input_data.thread_id}_{COLLECTION_NAME}",
            "gemini_api": gemini,
            "allowed_call_count": 2,
            "expired_call_count": 0
        }, thread):
            node_name = next(iter(event.keys()))
            yield f"data: {node_name}\n\n"
        
    return StreamingResponse(event_stream(), media_type="text/event-stream")
    

@app.post("/ragAnswer")
async def rag_answer(thread: Thread):
    thread = {"configurable": {"thread_id": thread.thread_id}}
    def event_stream():
        final_state = rag_graph.get_state(thread)
        answer = final_state.values.get('answer')
        yield f"{answer}"
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
    async def event_stream():
        final_state = medical_insights_graph.get_state(thread)
        medical_report = final_state.values.get('medical_report')
        yield f"{medical_report}"
        
    return StreamingResponse(event_stream(), media_type="text/event-stream")



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
async def input_query_for_image(thread_id:str=Form(...),query: str = Form(...)):

    async def resume_graph(thread_id:str, query: str):
        thread = {"configurable": {"thread_id": thread_id}}
        vision_graph.update_state(thread,{"query":query},as_node="enter query")
        vision_graph.invoke(None, thread)

    try:
        await resume_graph(thread_id,query)
        return {"Graph was resumed and query was taken into account. Check formed answer"}
    except Exception as e:
        return {f"graph failed, error: {e}"}
    
    
@app.post("/vision-answer/")
async def process_image_answer(thread_id: str = Form(...)):


    thread = {"configurable": {"thread_id": thread_id}}
    async def event_stream():
            final_state = vision_graph.get_state(thread)
            answer = final_state.values.get('answer')
            yield f"{answer}"
        
    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/vision-feedback")
async def process_vision_feedback(thread_id: str = Form(...), feedback: str= Form(...)):

    async def resume_graph(thread_id:str, feedback: str):
        thread = {"configurable": {"thread_id": thread_id}}
        vision_graph.update_state(thread, {"feedback":feedback}, as_node="human feedback")
        vision_graph.invoke(None,thread)

    try:
        await resume_graph(thread_id,feedback)
        return {"Graph was resumed and feedback was taken into account."}
    except Exception as e:
        return {f"graph failed, error: {e}"} 



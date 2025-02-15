import operator
from typing import List, TypedDict, Annotated, Optional, Dict, Literal
from pydantic import  BaseModel, Field
from langgraph.graph import MessagesState
from langgraph.graph import START,END,StateGraph, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.constants import Send
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
import os

from google.oauth2 import service_account
import base64
import json

encoded_creds = os.getenv("GOOGLE_CREDENTIALS_BASE64")
# Decode and load the JSON credentials
try:
    decoded_creds = base64.b64decode(encoded_creds).decode("utf-8")
    service_account_info = json.loads(decoded_creds)
    credentials = service_account.Credentials.from_service_account_info(service_account_info)
except Exception as e:
    raise ValueError("Failed to decode credentials") from e

class OverAllState(TypedDict):
    files: List
    medical_insights: Annotated[List, operator.add]
    medical_report: str
    human_feedback: str

class DocumentPages(TypedDict):
    file: tuple
    page_contents: List

class MedicalValues(BaseModel):
    test: str
    value: str

class Report(BaseModel):
    abnormal_values : Optional[List[MedicalValues]] = Field(default_factory=list)


detail_extractor_sys_mssg="""You are an expert in analyzing medical data, extract main medical details about a patient from the given context, 
do INCLUDE lab values, tests conducted, normal ranges. Do NOT include patient name, recommendations etc.
- Include Key Findings
- Avoid Normal Details until it is very important
- extract the most relevant medical details and summarize them significantly.  
- Focus on **clarity, accuracy, and readability** when presenting extracted insights.
- Return **No Medical Details Found** in case given docs are not some patient report.


"""


report_builder_sys_mssg = """
You are an expert medical report writer.

Your task is to:

1. **Format all text strictly in Markdown syntax** for clarity and readability.
2. Always **begin the report with "## Extracted Medical Report"**.
3. Ensure the content is **accurate, structured, and concise**, emphasizing **practical guidance**.
4. Focus on summarizing key **abnormal** findings (values outside the normal range).
5. **DO NOT** include normal values, only report significant deviations.
6. Maintain a **concise and professional tone** suitable for medical documentation.
7. The summary **must be a single paragraph** and strictly follow **Markdown** syntax.

## Extracted Medical Report  

- **Test1 Name [Test1 Value]**  
- **Test2 Name [Test2 Value]**  
- **Test3 Name [Test3 Value]**  
- **Test4 Name [Test4 Value]**  

### **✅ Summary:**  
Brief single-paragraph summary of abnormalities, ensuring anonymity of the patient.
"""


# report_template = """
# ## Extracted Medical Report

# {test_lines}

# #### **✅ Summary:**
# {summary}
# """

def process_documents(state:OverAllState):
    return {"files": state["files"]}

def should_trigger_edge_for_insights(state: OverAllState):
    return [Send("extract medical insights", {"file":(file_id,file)}) for file_id,file in state["files"]]

def extract_medical_insights(state:DocumentPages):
    file_id,file = state["file"]
    page_contents=[]

    for entry in file:
        page_contents.append(entry.page_content)

    sys_prompt = [SystemMessage(content=detail_extractor_sys_mssg)]
    llm_gemini = ChatGoogleGenerativeAI(api_key=os.getenv("GOOGLE_API_KEY"),
                                model="gemini-2.0-flash-exp",
                                credentials=credentials,
                                )
    
    
    response = llm_gemini.invoke(sys_prompt+[HumanMessage(content=f"Extract Relevant medical insights from {page_contents}")])
    

    return {"medical_insights":[response]}

def draft_report(state:OverAllState):
    medical_insights = [insight for insight in state["medical_insights"] if insight != "No Medical Details Found"]

    
    sys_prompt =  [SystemMessage(content=report_builder_sys_mssg)]
    # llm_groq = ChatGroq(
    #     model="llama-3.1-8b-instant",
    #     temperature=0,
    #     max_tokens=None,
    #     api_key=os.getenv("GROQ_API_KEY")
    # )
    llm_gemini = ChatGoogleGenerativeAI(api_key=os.getenv("GOOGLE_API_KEY"),
                                model="gemini-2.0-flash-exp",
                                credentials=credentials
                                )
    # structured_llm = llm_groq.with_structured_output(Report)

    # groq_res = structured_llm.invoke([HumanMessage(content=f"Draft according to the output, {state["medical_insights"]}")])
    
    # gemini_res = llm_gemini.invoke([SystemMessage(
    #     content="Frame you answer as per user request, respond with 'No medical summary' if no relevant medical insight was provided.")]+[HumanMessage(
    #     content=f"Draft a short and brief summary based on given insights, maintain anonymity of patient: {medical_insights}")])

    gemini_res = llm_gemini.invoke([SystemMessage(
        content=report_builder_sys_mssg)]+[HumanMessage(content=f"Maintain anonymity of patient, can mention age and sex, reply with 'No medical summary' if nothing relevant to medical report is found: {medical_insights}")])
    # gemini_res = llm_gemini.invoke([SystemMessage(
    #     content="Frame you answer as per user request, respond with 'No medical summary' if no relevant medical insight was provided. Output as strictly markdown!")]+[HumanMessage(
    #     content=f"Draft a short and brief summary in 1 para along with abnormal values based on given insights, maintain anonymity of patient: {medical_insights}")])
    

    if "No medical summary" in gemini_res.content:
        final_report ="No medical summary"
    else:
        final_report=gemini_res.content

    # else:
    #     test_lines = "  \n".join(
    #         "- **{} [{}]**".format(item.test, item.value)
    #         for item in groq_res.abnormal_values
    #     )

    #     summary_text = gemini_res.content.strip()

    #     final_report = (
    #         "## Extracted Medical Report  \n" +
    #         test_lines +
    #         "  \n#### **✅ Summary:**  \n" +
    #         summary_text
    #     )
        
    # res = [groq_res,gemini_res.content]

    return {"medical_report": final_report}


builder = StateGraph(OverAllState)
builder.add_node("process files",process_documents)
builder.add_node("extract medical insights", extract_medical_insights)
builder.add_node("draft report", draft_report)


builder.add_edge(START,"process files")
builder.add_conditional_edges(
    "process files",
    should_trigger_edge_for_insights,
    ["extract medical insights"]
)
builder.add_edge("extract medical insights","draft report")
builder.add_edge("draft report",END)


memory = MemorySaver()
medical_insights_graph = builder.compile(
    checkpointer=memory,
)


def trigger_medical_insights_extraction(uploaded_files, thread_id):
    thread = {"configurable": {"thread_id":thread_id}}
    files=[]
    for file_id,uploaded_file in enumerate(uploaded_files):
        with open(uploaded_file.filename, "wb") as f:
            f.write(uploaded_file.read())
        # Load and split the PDF into pages.
        loader = PyPDFLoader(uploaded_file.filename)
        pages = loader.load_and_split()
        files.append((file_id,pages))

    medical_insights_graph.invoke({"files":files}, thread)
    medical_report = medical_insights_graph.get_state(thread).values.get("medical_report")

    return medical_report
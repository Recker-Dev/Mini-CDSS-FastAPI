import operator
from typing import List, TypedDict, Annotated, Optional, Dict, Literal
from pydantic import  BaseModel, Field
from langgraph.graph import MessagesState
from langgraph.graph import START,END,StateGraph, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import Send
from langchain_community.document_loaders import PyPDFLoader

import os
from .llm import chat, chat_stream, sys, user
from .prompts import load_prompt


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


# Prompts live in prompts/*.md — see config/prompts.py.
detail_extractor_sys_mssg = load_prompt("medical_detail_extractor")
report_builder_sys_mssg = load_prompt("medical_report_builder")


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

    response = chat([
        sys(detail_extractor_sys_mssg),
        user(f"Extract Relevant medical insights from {page_contents}"),
    ])

    return {"medical_insights":[response]}

def stream_medical_report(medical_insights):
    """Stream the extracted-medical-report prose from the gathered insights."""
    insights = [insight for insight in medical_insights if insight != "No Medical Details Found"]
    yield from chat_stream([
        sys(report_builder_sys_mssg),
        user(f"Maintain anonymity of patient, can mention age and sex, reply with 'No medical summary' if nothing relevant to medical report is found: {insights}"),
    ], max_tokens=None)


# The graph gathers the structured insights; the report prose is streamed on
# demand from the endpoint via stream_medical_report.
builder = StateGraph(OverAllState)
builder.add_node("process files",process_documents)
builder.add_node("extract medical insights", extract_medical_insights)


builder.add_edge(START,"process files")
builder.add_conditional_edges(
    "process files",
    should_trigger_edge_for_insights,
    ["extract medical insights"]
)
builder.add_edge("extract medical insights", END)


memory = MemorySaver()
medical_insights_graph = builder.compile(
    checkpointer=memory,
)
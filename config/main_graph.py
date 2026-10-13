from langgraph.graph import START,END,StateGraph, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import Send
from .hugging_face_ner import  ner_extractor
from .llm import chat, chat_stream, structured, sys, user
from .prompts import load_prompt
from .search import web_search

from dotenv import load_dotenv
import os

load_dotenv()






import operator
from typing import List, TypedDict, Annotated, Optional, Dict
from pydantic import  BaseModel, Field


class NERReport(BaseModel):
    age: str = Field(..., description="The patient's age.")
    sex: str = Field(..., description="The patient's sex.")
    presenting_complaint: str = Field(
        ...,
        description="The primary complaint or issue reported by the patient."
    )
    history: Optional[str] = Field(
        None, 
        description="List of past medical history or conditions."
    )
    vital_signs: Optional[Dict[str,str]] = Field(
        None,
        description="Vital signs observed, including key metrics like heart rate, blood pressure, and temperature."
    )
    laboratory_values: Optional[Dict[str, str]] = Field(
        None,
        description=(
            "Key laboratory findings with test names as keys and results as values. "
        )
    )
    
    extra_summary: Optional[str] = Field(
        None, description="Additional relevant observations that do not fit into other categories."
    )

class PrelimDiagnosis(BaseModel):
    diagnosis: str = Field(..., description="Diagnosed disease")
    disease_description: str = Field(..., description="A summary about the disease")
    reason: str = Field(..., description="Thinking Behind diagnosis")
    recommendations: List[str] = Field(
    ...,
    description = "Certain Recommendations for the diagnosed outcome"
    )

class Diagnoses(BaseModel):
    list_of_diags :  List[PrelimDiagnosis]


class OverAllState(TypedDict):
    initial_summary: str
    medical_report: str
    diagnosis_count: int
    web_search_pro: bool

    diseases: List[str]

    post_ner_data: NERReport
    diagnoses:List[PrelimDiagnosis]
    best_practises: Annotated[List, operator.add]

    human_prelim_feedback: str

    ner_report:str
    prelim_report: str
    best_practise_report: str

class WebSearchState(BaseModel):
    disease: str
    web_search_pro: bool = False


# Prompts live in prompts/*.md — see config/prompts.py.
ner_validation_agent_sys_instruction = load_prompt("ner_validation")
ner_report_writer_agent_sys_instruction = load_prompt("ner_report_writer")
prelim_diagnosis_sys_instructions = load_prompt("prelim_diagnosis")
prelim_report_writer_agent_sys_instruction = load_prompt("prelim_report_writer")



def ner_extraction_validator(state: OverAllState):
    """Perform NER Extraction on medical data."""
    initial_summary = state["initial_summary"]
    medical_report = state.get("medical_report","")

    summarized_report = chat([
        sys(load_prompt("summarizer")),
        user(f"Here is the intial-report of the patient: {initial_summary} and the medical extracts (if any) drawn out {medical_report}"),
    ])

    # Optional Hugging Face NER augmentation; skips cleanly if HF_TOKEN is unset.
    try:
        ner_report = ner_extractor(summarized_report)
    except Exception:
        ner_report = ""

    ## Enforce the structured output
    post_ner_data = structured([
        sys(ner_validation_agent_sys_instruction),
        user(f"Validate the NER Report made by hugging face model (if any) : {ner_report}, on the input text of: {summarized_report}"),
    ], NERReport)

    return {"post_ner_data":post_ner_data}

def stream_ner_report(post_ner_data):
    """Stream the NER report prose (markdown), token-by-token, from structured data."""
    yield from chat_stream([
        sys(ner_report_writer_agent_sys_instruction),
        user(f"Use this following extract to draft a report: {post_ner_data}"),
    ], max_tokens=None)

def prelim_diagnosis_node(state: OverAllState):
    """Perform Prelim Diagnosis on the Patient Summary"""
    post_ner_data = state["post_ner_data"]

    ## Enforce Output
    diagnoses = structured([
        sys(prelim_diagnosis_sys_instructions.format(diagnosis_count= state["diagnosis_count"],
                                                     human_prelim_feedback= state.get("human_prelim_feedback",""))),
        user(f"Patient Data is as follows: {post_ner_data}. Give back relevant diagnoses."),
    ], Diagnoses)

    return {"diagnoses":diagnoses}
    
def prelim_human_feedback(state: OverAllState):
    """No-op node that should be interrupted on"""
    pass

def should_continue_edge_for_prelim(state: OverAllState):
    """Return the next node for Prelim Diagnosis Step to be executed"""

    human_prelim_feedback = state.get("human_prelim_feedback",None)

    if human_prelim_feedback:
        return "perform preliminary diagnosis node"
    
    return "extract diseases"

def stream_prelim_report(diagnoses):
    """Stream the Preliminary Diagnosis report prose from the structured diagnoses."""
    yield from chat_stream([
        sys(prelim_report_writer_agent_sys_instruction),
        user(f"Draft the Prelimn Report, context: {diagnoses.list_of_diags}"),
    ], max_tokens=None)

def extract_diseases(state:OverAllState):
    """Extract the diseases and updates the state"""
    list_of_diags = state["diagnoses"].list_of_diags
    diseases = [d.diagnosis for d in list_of_diags]
    return {"diseases":diseases}


def trigger_web_search(state:OverAllState):
    """Use Send() API to dig for best practises per disease"""
    pro = state.get("web_search_pro", False)
    return [Send("search web for best practises", {"disease": d, "web_search_pro": pro})
            for d in state["diseases"]]

def search_web_best_pracs(state: WebSearchState):
    """Retrieve best-practice sources per disease (SearXNG primary, Tavily fallback).

    The query avoids the word "best" (it makes metasearch return dictionary hits)
    and leans on clinical phrasing instead.
    """
    disease = state["disease"]
    query = f"{disease} treatment guidelines clinical management"
    results = web_search(query, max_results=5, pro=state.get("web_search_pro", False))

    if not results:
        return {"best_practises": [{
            "disease": disease,
            "url": "",
            "content": "No web results found (search provider unavailable or returned nothing).",
        }]}

    best_pracs = [
        {"disease": disease, "url": r["url"], "content": r["content"]}
        for r in results
    ]
    return {"best_practises": best_pracs}

best_pracs_section_sys_instruction = load_prompt("best_practices_section")


def stream_bestprac_report(best_practises):
    """Stream the Best Practises report, one disease section at a time.

    Each disease's section is generated independently (small, so it never hits the
    token ceiling) and the sections run concurrently across the server's parallel
    slots. Sections are yielded in order as they become ready.
    """
    import concurrent.futures
    from collections import OrderedDict

    # Group the gathered sources by disease, preserving first-seen order.
    by_disease = OrderedDict()
    for item in best_practises:
        by_disease.setdefault(item["disease"], []).append(item)

    def build_section(sources):
        return chat([
            sys(best_pracs_section_sys_instruction),
            user(f"Write the section for this disease from these sources: {sources}"),
        ], max_tokens=None)

    yield "## Best Practices Report\n\n"

    diseases = list(by_disease.keys())
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        futures = {d: ex.submit(build_section, by_disease[d]) for d in diseases}
        # Yield in disease order; concurrency means later sections are usually
        # already done by the time we reach them.
        for d in diseases:
            yield futures[d].result()
            yield "\n\n"


# The graph produces the *structured* work (NER data, diagnoses, web sources).
# The prose reports are streamed on demand from the endpoints via the
# stream_* helpers above, so they appear token-by-token and can be fetched
# concurrently.
builder = StateGraph(OverAllState)
builder.add_node("perform and validate NER extractions", ner_extraction_validator)
builder.add_node("perform preliminary diagnosis node", prelim_diagnosis_node)
builder.add_node("prelim human feedback node", prelim_human_feedback)
builder.add_node("extract diseases", extract_diseases)
builder.add_node("search web for best practises", search_web_best_pracs)

builder.add_edge(START, "perform and validate NER extractions")
builder.add_edge("perform and validate NER extractions", "perform preliminary diagnosis node")
builder.add_edge("perform preliminary diagnosis node", "prelim human feedback node")
builder.add_conditional_edges(
    "prelim human feedback node",
    should_continue_edge_for_prelim,
    ["perform preliminary diagnosis node", "extract diseases"]
)
builder.add_conditional_edges(
    "extract diseases",
    trigger_web_search,
    ["search web for best practises"]
)
builder.add_edge("search web for best practises", END)

memory = MemorySaver()

graph = builder.compile(
    interrupt_before=["prelim human feedback node"],
    checkpointer=memory,
)


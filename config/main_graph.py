from langgraph.graph import START,END,StateGraph, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.constants import Send
from .hugging_face_ner import  ner_extractor
from .credentials import creds

from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from tavily import TavilyClient
from dotenv import load_dotenv
import os

load_dotenv()

# from llm import llm_gemini,llm_groq
# from main_graph_tools import tavily_client






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


ner_validation_agent_sys_instruction ="""
You are an expert NER (Named Entity Recognition) agent specialized in extracting medical entities from text. 
Your task is to accurately identify and label entities such as diseases, symptoms, treatments, medications, 
anatomical locations, lab tests, and other medical terms. You are provided with both user text and a Hugging Face 
domain expert model output. Your goal is to refine, validate, and correct the extracted entities for medical accuracy.

### **Instructions:**
1. **Accurate Entity Recognition**  
   - Identify and extract only relevant medical entities with high precision.  
   - Ensure alignment with standard medical ontologies such as **UMLS** or **SNOMED**.  

2. **Context-Aware Classification**  
   - Categorize entities correctly into sections such as **Symptoms, Diagnoses, Treatments, Medications, 
     Vital Signs, and Lab Tests** based on context.  
   - Avoid misclassification—each entity must belong to the appropriate category.  

3. **Validation and Error Correction**  
   - Cross-check extracted entities against both the **user-provided text** and the **Hugging Face model output**.  
   - Fix any inaccuracies, misclassifications, or missing entities.  
   - Ensure no entity is duplicated or incorrectly tagged.  

4. **No Overlapping or Redundant Entities**  
   - Each entity should be placed in **only one relevant section**—no duplication across categories.  
   - Example: A value should not appear in both **Vital Signs** and **Lab Tests**—place it in the most 
     appropriate section.  

5. **Extra Summary Criteria**
    - To be provided only when any details is not enclosed by the model class. 
    - Should not repeat values of from **Vital Signs** or **Lab Tests**
    - Must be unique.

"""


ner_report_writer_agent_sys_instruction = """You are an expert NER Report writer on medical data.
Your task is to:
1. Format all text using Markdown syntax for clarity and readability, begin with ## Patient Initial Report always!
2. Ensure medical details are accurate and well-structured.
3. Use bullet points for clarity.
4. Follow a consistent section order as in the example.
5. Do not include unnecessary information—only what is extracted from NER.

Sample Format:
**Patient Details**  
- ...
- ...

**Presenting Complaint**  
- ...
- ...

**Vital Signs**  
- ... 
- ...

**Laboratory Values**  
- ...
- ...

**Treatment Initiated**  
- ... 

**Extra Details**
- ...
"""

prelim_diagnosis_sys_instructions = """You are a clinical decision assistant. Your job is to help
clinicians and healthcare provider with {diagnosis_count} unique but relevant diagnosis. You will be
provided with relevant patient details, such as age,sex,complaints,history,vital signs, lab values etc.
Based on the provided details come up with the most relevant diagnosis.

Examine any editorial feedback that has been optionally provided to guide the diagnosis:
{human_prelim_feedback}

"""

prelim_report_writer_agent_sys_instruction = """You are an expert Prelim Report writer on medical data.  
Your task is to:  
1. Format all text using **Markdown syntax** for clarity and readability, always begin with ## Prelim Diagnosis Report always!.  
2. Ensure medical details are **accurate, well-structured, and concise**.  
3. Use **clear section sub-headers, bullet points and numbering** for readability.  
4. Maintain a **consistent order** for each diagnosis:  
   - **Diagnosis**  
   - **Disease Description**  
   - **Clinical Reasoning**  
   - **Recommendations**  
5. Include only the **necessary extracted medical details**, avoiding redundant information.  

**Sample Format:**  
## Prelim Diagnosis Report  

### **Diagnosis Name**  
**📌 Disease Description:**  
- ...  

**🧠 Clinical Reasoning:**  
- ...  

**✅ Recommendations:**  
- ...  
- ...  
- ...  

You are provided with a list of Preliminary Diagnoses, do this for each, number them accordingly.
"""

best_pracs_report_writer_agent_sys_instruction = """You are an expert **Best Practices Medical Report Writer**.  
Your task is to:  

1. Format all text using **Markdown syntax** for clarity and readability, always beginning with `## Best Practices Report`.  
2. Ensure the content is **accurate, structured, and concise**, emphasizing **practical guidance**.  
3. Use **clear sub-headers, bullet points, and numbering** to enhance readability.  
4. If a disease has multiple sources, **group them together**, mention the **total number of sources** in brackets, and list them in the given format.  
5. Include only **relevant and actionable medical insights**, avoiding unnecessary details.  

### **Example Format:**  

## Best Practices Report  

### **🩺 Disease Name** [2 sources]  
- **🔗 Source:** [URL]  
  - **📖 Summary:** ....  

- **🔗 Source:** [URL]  
  - **📖 Summary:** ....  

Repeat this format for each disease in the provided data.  
"""



def ner_extraction_validator(state: OverAllState):
    """Perform NER Extraction on medical data."""
    initial_summary = state["initial_summary"]
    medical_report = state.get("medical_report","")

    llm_gemini = ChatGoogleGenerativeAI(api_key=os.getenv("GOOGLE_API_KEY"),
                            model="gemini-2.0-flash-exp",
                            credentials=creds
                            )

    summarized_report = llm_gemini.invoke(
        [SystemMessage(content="Your task is to summarize all the content that is given to you. Leave no medical details out, no matter how small. Output is in text-string format and not markdown, avoid special characters.")]+ 
        [HumanMessage(content= f"Here is the intial-report of the patient: {initial_summary} and the medical extracts (if any) drawn out {medical_report}")]
    )

    # tagged_tokens, unique_tags = process_ner_output(initial_summary+" "+medical_report)
    # report = generate_clean_ner_report(tagged_tokens, unique_tags)
    try:
        ner_report = ner_extractor(summarized_report.content)
    except:
        ner_report = ""

    llm_groq = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        max_tokens=None,
        api_key=os.getenv("GROQ_API_KEY")
    )

    ## Enforce the structured output
    structured_llm = llm_groq.with_structured_output(NERReport)

    # sys_mssg = ner_validation_agent_sys_instruction

    post_ner_data = structured_llm.invoke(
        [SystemMessage(content=ner_validation_agent_sys_instruction)]+[HumanMessage(
            content=f"Validate the NER Report made by hugging face model (if any) : {ner_report}, on the input text of: {summarized_report.content}")]
        )

    
    return {"post_ner_data":post_ner_data}

def ner_report_builder_node(state: OverAllState):

    """Create a NER Report in Mardown Format. """
    post_ner_data = state["post_ner_data"]

    llm_gemini = ChatGoogleGenerativeAI(api_key=os.getenv("GOOGLE_API_KEY"),
                                model="gemini-2.0-flash-exp",
                                credentials=creds
                                )

    report = llm_gemini.invoke(
        [SystemMessage(content=ner_report_writer_agent_sys_instruction)]+ [HumanMessage(content= f"Use this following extract to draft a report: {post_ner_data}")]
    )

    return {"ner_report": report.content}

def prelim_diagnosis_node(state: OverAllState):
    """Perform Prelim Diagnosis on the Patient Summary"""
    post_ner_data = state["post_ner_data"]

    # print(F"FEEDBACK PROVIDED AS OF NOW:{state.get("human_prelim_feedback","")}")

    ## Enfore Output
    llm_gemini = ChatGoogleGenerativeAI(api_key=os.getenv("GOOGLE_API_KEY"),
                                model="gemini-2.0-flash-exp",
                                credentials=creds
                                )

    structured_llm = llm_gemini.with_structured_output(Diagnoses)

    diagnoses = structured_llm.invoke(
        [SystemMessage(content=prelim_diagnosis_sys_instructions.format(diagnosis_count= state["diagnosis_count"], 
                                                                        human_prelim_feedback= state.get("human_prelim_feedback","")))]
        + [HumanMessage(
            content=f"Patient Data is as follows: {post_ner_data}. Give back relevant diagnoses.")]
    )

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

def prelim_report_builder_node(state:OverAllState):
    """Create Prelim Report in NER format"""
    llm_gemini = ChatGoogleGenerativeAI(api_key=os.getenv("GOOGLE_API_KEY"),
                                model="gemini-2.0-flash-exp",
                                credentials=creds
                                )
    
    prelim_report = llm_gemini.invoke(
        [SystemMessage(content=prelim_report_writer_agent_sys_instruction)]
         + [HumanMessage(content=f"Draft the Prelimn Report, context: {state['diagnoses'].list_of_diags}")]
    )

    return {"prelim_report": prelim_report.content}

def extract_diseases(state:OverAllState):
    """Extract the diseases and updates the state"""
    list_of_diags = state["diagnoses"].list_of_diags
    diseases = [d.diagnosis for d in list_of_diags]
    return {"diseases":diseases}


def trigger_web_search(state:OverAllState):
    """Use Send() API to dig for best practises per disease"""
    return [Send("search web for best practises", {"disease":d}) for d in state["diseases"]]

def search_web_best_pracs(state: WebSearchState):
    """Retrieve docs from web-search"""
    tavily_client=TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

    # Search
    res = tavily_client.search(f"What are the best medical practises for: {state['disease']}")
    best_pracs =[]
    for doc in res["results"]:
        url = doc["url"]
        content = doc["content"]
        best_pracs.append(
            {
            "disease":state["disease"],
            "url": url,
            "content": content
            })

    return {"best_practises":best_pracs}

def best_pracs_report_builder_node(state: OverAllState):
    """Best Practises Report Writing Node"""
    llm_gemini = ChatGoogleGenerativeAI(api_key=os.getenv("GOOGLE_API_KEY"),
                                model="gemini-2.0-flash-exp",
                                credentials=creds
                                )
    
    best_practise_report = llm_gemini.invoke(
        [SystemMessage(content=best_pracs_report_writer_agent_sys_instruction)]
         + [HumanMessage(content=f"Draft the Best Practises Report, context: {state['best_practises']}")]
    )

    return {"best_practise_report": best_practise_report.content}


builder = StateGraph(OverAllState)
builder.add_node("perform and validate NER extractions", ner_extraction_validator)
builder.add_node("write NER report",ner_report_builder_node)
builder.add_node("perform preliminary diagnosis node", prelim_diagnosis_node)
builder.add_node("prelim human feedback node", prelim_human_feedback)
builder.add_node("write prelim report", prelim_report_builder_node)
builder.add_node("extract diseases", extract_diseases)
builder.add_node("search web for best practises", search_web_best_pracs)
builder.add_node("write best practises report", best_pracs_report_builder_node)

builder.add_edge(START, "perform and validate NER extractions")
builder.add_edge("perform and validate NER extractions", "write NER report")
builder.add_edge("perform and validate NER extractions", "perform preliminary diagnosis node")
builder.add_edge("perform preliminary diagnosis node","write prelim report" )
builder.add_edge("write prelim report", "prelim human feedback node")
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
builder.add_edge("search web for best practises","write best practises report")

memory = MemorySaver()

graph = builder.compile(
    interrupt_before=["prelim human feedback node"],
    checkpointer=memory,
)


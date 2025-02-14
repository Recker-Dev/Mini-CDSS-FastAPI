from typing import List, TypedDict,Literal
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import  BaseModel, Field
from langgraph.graph import START,END,StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
import os

class Query(BaseModel):
    query:str

class QueryList(BaseModel):
    queries: List[Query]

class Relevance(BaseModel):
    isRelevant:Literal["yes","no"]
    suggestion: str
    
class OverAllState(TypedDict):
    question: str
    collection_path: str
    gemini_api:str
    queries: QueryList
    max_queries: int
    docs_retrieved: List
    relevance: Literal["yes","no"]
    suggestion: str
    allowed_call_count:int 
    expired_call_count:int
    answer: str

frame_queries_sys_instruction="""
You are an expert in data retrieval and query generation for vector databases.
Given a user question, your job is to frame {max_queries} optimized queries that maximize relevant information retrieval.

Guidelines:
- Ensure the queries are semantically diverse while staying relevant to the original question.
- Avoid redundancy; each query should retrieve distinct but related information.
- If necessary, reformulate the queries to handle ambiguous or incomplete user inputs.
- Structure the queries to optimize retrieval efficiency in vector search.

In case your queries were not relevant, you will be given a suggestion.
Following is some editorial suggestion (if any): {suggestion}

"""
relevance_checker_sys_instruction = """
You are an expert in evaluating the relevance of retrieved documents to a given user question.

### **Input:**
- **User Question:** {question}  
- **Retrieved Documents:** {docs}  

### **Task:**  
Analyze the retrieved documents in the context of the user's question. Determine whether they contain relevant information that directly addresses the query.  

### **Expected Output:**  
You must respond with:  
- **"Yes"** → If at least one document is relevant.  
- **"No"** → If none of the documents are relevant.  

Additionally, if the response is **"No"**, provide a detailed yet concise **suggestion** explaining why the retrieved documents were not relevant and how to improve future retrievals.
Starting with 'Previously retrieved documents...'  
If the response is **"Yes"**, return **"No suggestion needed"** instead.

"""

answer_generation_sys_instruction = """
You are an expert in generating well-structured reports based on a given user question and retrieved relevant documents.

### **Input:**
- **User Question:** {question}  
- **Retrieved Documents:** {docs}  

### **Task:**
Analyze the retrieved documents in the context of the user's question. Extract key insights, summarize relevant details, and generate a clear, concise, and well-structured answer that directly addresses the question.

### **Guidelines:**
- Ensure the report is **coherent, informative, and to the point**.  
- Maintain a **professional and neutral tone**.  
- If the documents do not provide a direct answer, indicate the **limitations** and suggest possible areas for further research.  

### **Expected Output:**
A structured and well-written report that effectively summarizes the relevant information from the provided documents in response to the user's question.
Reply like a chatbot, strictly in markdown format, but ONLY markdown not in "```markdown ```"

"""

def frame_queries(state: OverAllState):
    question = state["question"]
    
    sys_prompt = [SystemMessage(content=frame_queries_sys_instruction.format(
        max_queries= state["max_queries"],
        suggestion = state.get("suggestion","")
    ))]
    llm_gemini = ChatGoogleGenerativeAI(api_key=os.getenv("GOOGLE_API_KEY"),
                                model="gemini-2.0-flash-exp",
                                )

    structured_llm = llm_gemini.with_structured_output(QueryList)

    response = structured_llm.invoke(sys_prompt+[HumanMessage(
        content=f"User question is as follows: {question}"
    )])

    return {"queries":response,
            "expired_call_count":state["expired_call_count"]+1}

def retrieve_docs(state: OverAllState):
    db_path = ".chroma"
    queries_list = state["queries"].queries
    COLLECTION_NAME =state["collection_path"]

    queries_list = [entry.query for entry in queries_list]

    chroma_client = chromadb.PersistentClient(path=db_path)
    google_ef  = embedding_functions.GoogleGenerativeAiEmbeddingFunction(api_key=state["gemini_api"])
    collection=chroma_client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=google_ef)


    results = collection.query(query_texts=queries_list,n_results=5)
    docs = results["documents"]

    # print(f"Try Number: {state["expired_call_count"]-1}, docs lenght: {len(docs_cummulative)}")
    return {"docs_retrieved":docs}

def check_relevance(state: OverAllState):
    if not state["docs_retrieved"]:
        return {"relevance": "no"}

    sys_prompt = [SystemMessage(content=relevance_checker_sys_instruction.format(
        question= state["question"],
        docs = state.get("docs_retrieved","")
    ))]
    llm_gemini = ChatGoogleGenerativeAI(api_key=os.getenv("GOOGLE_API_KEY"),
                                model="gemini-2.0-flash-exp",
                                )

    structured_llm = llm_gemini.with_structured_output(Relevance)

    response = structured_llm.invoke(sys_prompt+[HumanMessage(
        content=f"Check for relevance of the documents"
    )])
    # print(response)
    return {"relevance":response.isRelevant}

def should_trigger_edge_for_drafting(state: OverAllState):
    if state["relevance"]=="yes" or state["expired_call_count"]>state["allowed_call_count"]:
        # print("Should Draft answer now!")
        return "draft answer"
    
    # print("Retrying..")
    return "frame queries"


def draft_answer(state: OverAllState):
    # print("Drafted Answer")
    # print(state["relevance"])
    if state["relevance"]=="no":
        return {"answer":"No answer can be generated, documents related to question was not there in the vector database. Please re-try with a relevant query or upload relevant documents and try again."}
    
    sys_prompt= [SystemMessage(content=answer_generation_sys_instruction.format(
        question = state["question"],
        docs = state["docs_retrieved"]
    ))]
    llm_gemini = ChatGoogleGenerativeAI(api_key=os.getenv("GOOGLE_API_KEY"),
                                model="gemini-2.0-flash-exp",
                                )

    response = llm_gemini.invoke(sys_prompt+[HumanMessage(content="Draft the answer")])

    return {"answer":response.content}


builder = StateGraph(OverAllState)

builder.add_node("frame queries", frame_queries)
builder.add_node("retrieve docs", retrieve_docs)
builder.add_node("check relevance", check_relevance)
builder.add_node("draft answer", draft_answer )

builder.add_edge(START, "frame queries")
builder.add_edge("frame queries", "retrieve docs")
builder.add_edge("retrieve docs","check relevance")
builder.add_conditional_edges(
    "check relevance",
    should_trigger_edge_for_drafting,
    ["draft answer","frame queries"]
)
builder.add_edge("draft answer", END)
memory = MemorySaver()
rag_graph = builder.compile(checkpointer=memory)
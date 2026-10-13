from typing import List, TypedDict,Literal
from pydantic import  BaseModel, Field
from langgraph.graph import START,END,StateGraph
from langgraph.checkpoint.memory import MemorySaver
import chromadb
import os
from .llm import chat_stream, structured, sys, user, get_embedding_function
from .prompts import load_prompt

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
    queries: QueryList
    max_queries: int
    docs_retrieved: List
    relevance: Literal["yes","no"]
    suggestion: str
    allowed_call_count:int 
    expired_call_count:int
    answer: str

# Prompts live in prompts/*.md — see config/prompts.py.
frame_queries_sys_instruction = load_prompt("rag_frame_queries")
relevance_checker_sys_instruction = load_prompt("rag_relevance_checker")
answer_generation_sys_instruction = load_prompt("rag_answer_generation")

def frame_queries(state: OverAllState):
    question = state["question"]

    response = structured([
        sys(frame_queries_sys_instruction.format(
            max_queries=state["max_queries"],
            suggestion=state.get("suggestion", ""),
        )),
        user(f"User question is as follows: {question}"),
    ], QueryList)

    return {"queries":response,
            "expired_call_count":state["expired_call_count"]+1}

def retrieve_docs(state: OverAllState):
    db_path = ".chroma"
    queries_list = state["queries"].queries
    COLLECTION_NAME =state["collection_path"]

    queries_list = [entry.query for entry in queries_list]

    chroma_client = chromadb.PersistentClient(path=db_path)
    collection=chroma_client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=get_embedding_function())


    results = collection.query(query_texts=queries_list,n_results=5)
    docs = results["documents"]

    # print(f"Try Number: {state["expired_call_count"]-1}, docs lenght: {len(docs_cummulative)}")
    return {"docs_retrieved":docs}

def check_relevance(state: OverAllState):
    if not state["docs_retrieved"]:
        return {"relevance": "no"}

    response = structured([
        sys(relevance_checker_sys_instruction.format(
            question=state["question"],
            docs=state.get("docs_retrieved", ""),
        )),
        user("Check for relevance of the documents"),
    ], Relevance)
    # print(response)
    return {"relevance":response.isRelevant}

def should_trigger_edge_for_drafting(state: OverAllState):
    if state["relevance"]=="yes" or state["expired_call_count"]>state["allowed_call_count"]:
        # Retrieval is good enough (or we're out of retries); the answer is
        # streamed on demand from the endpoint via stream_rag_answer.
        return END

    # print("Retrying..")
    return "frame queries"


NO_ANSWER_MESSAGE = (
    "No answer can be generated, documents related to question was not there in the "
    "vector database. Please re-try with a relevant query or upload relevant documents "
    "and try again."
)


def stream_rag_answer(question, docs_retrieved, relevance):
    """Stream the RAG answer token-by-token from the retrieved docs."""
    if relevance == "no":
        yield NO_ANSWER_MESSAGE
        return

    yield from chat_stream([
        sys(answer_generation_sys_instruction.format(question=question, docs=docs_retrieved)),
        user("Draft the answer"),
    ], max_tokens=None)


builder = StateGraph(OverAllState)

builder.add_node("frame queries", frame_queries)
builder.add_node("retrieve docs", retrieve_docs)
builder.add_node("check relevance", check_relevance)

builder.add_edge(START, "frame queries")
builder.add_edge("frame queries", "retrieve docs")
builder.add_edge("retrieve docs","check relevance")
builder.add_conditional_edges(
    "check relevance",
    should_trigger_edge_for_drafting,
    [END, "frame queries"]
)
memory = MemorySaver()
rag_graph = builder.compile(checkpointer=memory)
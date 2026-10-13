from typing import Optional, TypedDict
from pydantic import BaseModel


# Define the Pydantic model

class Thread(BaseModel):
    thread_id:str

class  GraphInput(BaseModel):
    thread_id: str
    text: str
    diagnosis_count: str
    medical_report: str
    web_search_pro: bool = False  # True -> Tavily ("Search Pro"); False -> SearXNG

class PrelimInterrupt(BaseModel):
    thread_id: str
    human_feedback: Optional[str] = None ## Pass empty string for trigerring None

class APIInput(BaseModel):
    # Only Tavily (optional web search) is still used. gemini/groq are accepted
    # for backwards compatibility with older clients but ignored.
    tavily: Optional[str] = None
    gemini: Optional[str] = None
    groq: Optional[str] = None

class RagChat(BaseModel):
    thread_id: str
    question: str
    gemini: Optional[str] = None
    
class VisionInput(BaseModel):
    thread_id: str
    query: str

class VisionFeedback(BaseModel):
    thread_id: str
    feedback: str
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

class PrelimInterrupt(BaseModel):
    thread_id: str
    human_feedback: Optional[str] = None ## Pass empty string for trigerring None

class APIInput(BaseModel):
    gemini: str
    groq: str
    tavily: str

class RagChat(BaseModel):
    thread_id: str
    question: str
    gemini: Optional[str]  
    

from langchain_google_genai import ChatGoogleGenerativeAI
import os
from langgraph.graph import START,END,StateGraph, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from groq import  Groq
from google import genai
import base64
from google.genai import types
from .credentials import creds

from dotenv import load_dotenv

from typing_extensions import  TypedDict

class OverAllState(TypedDict):
    query: str
    base64_image: str
    llama_response: str
    gemini_response: str
    answer:str
    


load_dotenv()




system_instruction = '''
You are a prompt analyzer for medical or diagnostic inquiries.
Your task is to determine whether the user is asking for a detailed, structured answer or just a general response.

Instructions:
- If the user's prompt indicates a desire for a **detailed response** (e.g., diagnosis, comprehensive report, in-depth analysis), return "yes".
- If the user's prompt indicates a need for a **brief, general response** (e.g., simple query, identification), return "no".

Analyze the prompt and return your answer in the format: 
**yes**
or
**no**
'''
answer_writing_instruction ="""
You are an AI assistant that summarizes medical image analysis results concisely for users. 

Based on the given inputs, generate a **brief, user-friendly summary** of the most likely condition.
Output should be as markdown, divide properly in points and subpoints.

Constraints:
- Keep the response **to the point**.
- Avoid technical jargon; use **plain language**.
- Include details if mentioned so by the user.
- Suggest Home remedies, details about the issue and recommendations
- Frame a proper response, addressing to each of user's queries and sub-queries

"""


def process_image_llama(state: OverAllState):
    client = Groq(api_key=os.getenv("GROQ_API_KEY")) 
    query = state["query"]
    base64_image = state["base64_image"]

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"{query}"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        }
    ]

    chat_completion = client.chat.completions.create(
        messages=messages,
        model="llama-3.2-90b-vision-preview"
    )

    # Correct variable name used for response extraction.
    response = chat_completion.choices[0].message.content
    # print("Llama Response: ", response)
    # print("LLAMA TRIGGERED")
    return {"llama_response": response}
    # return {"llama_response": "got"}
    



def process_image_gemini(state: OverAllState):
    """Processes an image with Google Gemini Vision model."""
    query = state["query"]
    base64_image = state["base64_image"]

    # Decode base64 string to bytes
    image_bytes = base64.b64decode(base64_image)

    # Call Gemini Vision API
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    response = client.models.generate_content(
        model="gemini-2.0-flash-exp",
        contents=[
            query,
            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
        ],
    )

    # print("Gemini Response: ", response.text)
    # print("GEMINI TRIGGERED")
    return {"gemini_response": response.text}
    # return {"gemini_response": "got"}



def build_answer(state: OverAllState):
    # if (state["gemini_response"] or state["llama_response"] ) and (state["prompt_analyzer_response"] == "yes"):

    llm_gemini = ChatGoogleGenerativeAI(api_key=os.getenv("GOOGLE_API_KEY"),
                             model="gemini-2.0-flash-exp",
                             credentials=creds)
        
    response = llm_gemini.invoke([SystemMessage(content=answer_writing_instruction)]+ [HumanMessage(
        content=f"Produce an answer,based on\n QUERY: {state['query']}\n GEMINI RESPONSE:{state['gemini_response']}\n LLAMA REPONSE: {state['llama_response']}")])
        
    # print("ANSWER NODE TRIGGERED")
    return {"answer":response.content}


# Build a simple graph with one node.
builder = StateGraph(OverAllState)
builder.add_node("process image llama", process_image_llama)
# builder.add_node("prompt analyzer", prompt_analyzer)
builder.add_node("process image gemini",process_image_gemini)
builder.add_node("build answer", build_answer)

builder.add_edge(START, "process image llama")
# builder.add_edge(START,"prompt analyzer")
builder.add_edge(START, "process image gemini")
builder.add_edge("process image llama", "build answer")
builder.add_edge("process image gemini", "build answer")
# builder.add_edge("prompt analyzer", "build answer")
builder.add_edge("build answer", END)
memory = MemorySaver()
vision_graph = builder.compile(checkpointer=memory)

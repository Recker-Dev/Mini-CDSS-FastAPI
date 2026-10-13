import os
from langgraph.graph import START,END,StateGraph, MessagesState
from langgraph.checkpoint.memory import MemorySaver

from dotenv import load_dotenv

load_dotenv()


from typing_extensions import  TypedDict, Literal
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Vision is currently DISABLED (graceful early-exit).
#
# The original graph ran image understanding through Gemini Vision + Groq's
# Llama-4 vision model. The current local model (Qwen3.5) is text-only (no
# vision projector / mmproj), so it cannot process images.
#
# The graph structure below is kept fully intact so vision can be restored
# later: re-point the image nodes at a multimodal model and flip VISION_SUPPORTED
# to True. Until then, every node simply returns the message below and the graph
# exits cleanly instead of calling any remote service.
# ---------------------------------------------------------------------------
VISION_SUPPORTED = False

VISION_UNSUPPORTED_MESSAGE = (
    "Vision capabilities are not supported by the current local model. "
    "The configured model (Qwen3.5) is text-only. To enable image analysis, "
    "serve a vision-capable (multimodal) model and set VISION_SUPPORTED = True "
    "in config/vision_graph.py."
)

class OverAllState(TypedDict):
    query: str
    base64_image: str
    llama_response: str
    gemini_response: str
    answer:str
    feedback: str

class VisionModelDecisionEdgeOutput(BaseModel):
    outcome: Literal["retrigger image models", "rebuild answer", "end"]



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
Gemini Output: `{gemini_output}` 
AND
Llama Output:  `{llama_output}`

Constraints:
- Keep the response **to the point**.
- Avoid technical jargon; use **plain language**.
- Include details if mentioned so by the user.
- Suggest Home remedies, details about the issue and recommendations
- Frame a proper response, addressing to each of user's queries and sub-queries

"""


def get_user_query(state:OverAllState):
    pass

def process_image_llama(state: OverAllState):
    # Disabled: the local model has no vision. Restore the Groq Llama-4 vision
    # call here (see git history) once a multimodal backend is available.
    if not VISION_SUPPORTED:
        return {"llama_response": VISION_UNSUPPORTED_MESSAGE}




def process_image_gemini(state: OverAllState):
    """Processes an image with a vision model (disabled while text-only)."""
    # Disabled: restore the multimodal call here once VISION_SUPPORTED is True.
    if not VISION_SUPPORTED:
        return {"gemini_response": VISION_UNSUPPORTED_MESSAGE}




def build_answer(state: OverAllState):
    # Without vision there is nothing to summarize; surface the clear message.
    if not VISION_SUPPORTED:
        return {"answer": VISION_UNSUPPORTED_MESSAGE}


def human_feedback(state: OverAllState):
    pass

def should_trigger_feedback_edge(state: OverAllState):
    # Vision is disabled, so no retriggering is possible; end the graph cleanly.
    if not VISION_SUPPORTED:
        return END

    # (When vision is restored, bring back the feedback-routing LLM call here.)
    return END


# Build a simple graph with one node.
builder = StateGraph(OverAllState)
builder.add_node("enter query",get_user_query)
builder.add_node("process image llama", process_image_llama)
builder.add_node("process image gemini",process_image_gemini)
builder.add_node("build answer", build_answer)
builder.add_node("human feedback", human_feedback)

builder.add_edge(START, "enter query")
builder.add_edge("enter query", "process image llama")
builder.add_edge("enter query", "process image gemini")
builder.add_edge("process image llama", "build answer")
builder.add_edge("process image gemini", "build answer")
builder.add_edge("build answer","human feedback")
builder.add_conditional_edges(
    "human feedback",
    should_trigger_feedback_edge,
    ["process image llama","process image gemini","build answer", END]
)

memory = MemorySaver()
vision_graph = builder.compile(checkpointer=memory, interrupt_before=["human feedback","enter query"])
vision_graph

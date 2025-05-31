# main.py
from fastapi import FastAPI, Request
from pydantic import BaseModel
from langgraph_app import app_flow  # import compiled LangGraph app

app = FastAPI()

class ChatRequest(BaseModel):
    message: str
    state: dict

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    user_input = req.message
    state = req.state or {}  # Use incoming state
    
    # Update state with the new user query
    state["query"] = user_input
    state.setdefault("docs", [])
    state.setdefault("recipe", "")
    state.setdefault("result", "")
    state.setdefault("chat_history", [])
    state.setdefault("step_index", 0)
    state.setdefault("continue_chat", True)

    # Optionally append to chat_history here or in app_flow
    state["chat_history"].append(("user", user_input))
    
    new_state = app_flow.invoke(state)
    #state["query"] = user_input
    new_state = app_flow.invoke(state)
    return {"response": new_state["result"], "state": new_state}

from __future__ import annotations

from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from config import get_settings
from database import ConversationStore
from graph import ask_question, build_graph


@asynccontextmanager
async def lifespan(app: FastAPI):
    store = ConversationStore(get_settings().DATABASE_URL)
    try:
        store.initialize()
        if store.checkpointer is None:
            raise RuntimeError("PostgreSQL checkpointer initialization failed.")
        app.state.conversation_store = store
        app.state.workflow = build_graph(checkpointer=store.checkpointer)
    except Exception:
        store.close()
        raise
    try:
        yield
    finally:
        store.close()


app = FastAPI(title="Mini LLM", version="0.1.0", lifespan=lifespan)


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1)
    thread_id: str | None = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask")
def ask_question_api(payload: QuestionRequest, request: Request) -> dict:
    thread_id = payload.thread_id or str(uuid4())
    result = ask_question(
        payload.question,
        thread_id=thread_id,
        workflow=request.app.state.workflow,
    )
    request.app.state.conversation_store.save_conversation(
        thread_id,
        payload.question[:200],
    )
    return {
        "thread_id": thread_id,
        "answer": result.get("answer", ""),
        "cache_hit": result.get("cache_hit", False),
        "cache_type": result.get("cache_type", "none"),
        "llm_called": result.get("llm_called", False),
        "model": result.get("model", ""),
        "latency_ms": result.get("latency_ms", 0.0),
    }


@app.get("/conversations")
def list_conversations(request: Request) -> dict:
    return {"conversations": request.app.state.conversation_store.list_conversations()}


@app.get("/conversations/{thread_id}")
def get_conversation(thread_id: str, request: Request) -> dict:
    store: ConversationStore = request.app.state.conversation_store
    conversation = store.get_conversation(thread_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    snapshot = request.app.state.workflow.get_state(
        {"configurable": {"thread_id": thread_id}}
    )
    messages = [
        {
            "role": "assistant" if message.type == "ai" else "user",
            "content": message.content,
        }
        for message in snapshot.values.get("messages", [])
        if message.type in {"human", "ai"}
    ]
    return {**conversation, "messages": messages}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)

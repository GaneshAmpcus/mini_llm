from __future__ import annotations

from functools import lru_cache
from typing import Sequence

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from config import get_settings


@lru_cache(maxsize=1)
def get_chat_model() -> ChatGroq:
    settings = get_settings()
    if not settings.GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is required before invoking the Groq provider.")

    return ChatGroq(
        groq_api_key=settings.GROQ_API_KEY,
        model=settings.GROQ_MODEL,
        temperature=settings.LLM_TEMPERATURE,
        max_tokens=settings.LLM_MAX_TOKENS,
    )


def generate_answer(
    question: str,
    *,
    conversation_messages: Sequence[BaseMessage] | None = None,
    system_prompt: str | None = None,
) -> str:
    """Generate a response using the configured Groq chat model."""
    settings = get_settings()
    model = get_chat_model()
    final_prompt = system_prompt or (
        "You are a helpful technical assistant. Answer concise, accurate technical questions."
    )

    try:
        history = list(conversation_messages or [HumanMessage(content=question)])
        messages: Sequence[BaseMessage] = [SystemMessage(content=final_prompt), *history]
        response = model.invoke(messages)
        content = getattr(response, "content", str(response))
        answer = str(content).strip()
        if not answer:
            raise ValueError("Groq returned an empty response")
        return answer
    except Exception as exc:  # pragma: no cover - surfaced as runtime error
        raise RuntimeError(f"LLM generation failed with Groq: {exc}") from exc

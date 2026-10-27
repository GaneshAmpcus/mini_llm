from __future__ import annotations

from cache.exact_cache import ExactCache
from config import Settings, get_settings


def test_settings_defaults(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("GROQ_MODEL", "llama-3.1-8b-instant")
    settings = Settings()
    assert settings.GROQ_MODEL == "llama-3.1-8b-instant"
    assert settings.REDIS_URL == "redis://localhost:6379/0"
    assert settings.ENABLE_EXACT_CACHE is False


def test_exact_cache_key_is_deterministic():
    key_1 = ExactCache.build_cache_key(
        "What is LangGraph?",
        "llama-3.1-8b-instant",
        "v1",
        temperature=0.1,
        max_tokens=512,
    )
    key_2 = ExactCache.build_cache_key(
        "  what is langgraph?  ",
        "llama-3.1-8b-instant",
        "v1",
        temperature=0.1,
        max_tokens=512,
    )
    assert key_1 == key_2


def test_get_settings_uses_env(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "demo-key")
    monkeypatch.setenv("GROQ_MODEL", "llama-3.1-8b-instant")
    settings = get_settings()
    assert settings.GROQ_API_KEY == "demo-key"
    assert settings.GROQ_MODEL == "llama-3.1-8b-instant"


def test_graph_restores_messages_for_same_thread(monkeypatch):
    from langgraph.checkpoint.memory import MemorySaver

    import graph

    monkeypatch.setenv("GROQ_API_KEY", "demo-key")
    monkeypatch.setenv("GROQ_MODEL", "llama-3.1-8b-instant")
    monkeypatch.setenv("ENABLE_EXACT_CACHE", "false")
    monkeypatch.setenv("ENABLE_SEMANTIC_CACHE", "false")
    get_settings.cache_clear()

    calls = []

    def fake_generate_answer(question, *, conversation_messages=None):
        calls.append([message.type for message in conversation_messages])
        return f"answer: {question}"

    monkeypatch.setattr(graph, "generate_answer", fake_generate_answer)
    workflow = graph.build_graph(checkpointer=MemorySaver())
    graph.ask_question("What is LangGraph?", "test-thread", workflow)
    graph.ask_question("How does it keep state?", "test-thread", workflow)

    snapshot = workflow.get_state({"configurable": {"thread_id": "test-thread"}})
    assert [message.type for message in snapshot.values["messages"]] == [
        "human",
        "ai",
        "human",
        "ai",
    ]
    assert calls[1] == ["human", "ai", "human"]
    get_settings.cache_clear()

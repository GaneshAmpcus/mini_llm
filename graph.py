from __future__ import annotations

import logging
import time

from langchain_core.messages import AIMessage
from langgraph.graph import END, StateGraph

from cache.exact_cache import ExactCache
from cache.semantic_cache import SemanticCache
from config import get_settings
from llm.provider import generate_answer
from state import GraphState


logger = logging.getLogger(__name__)


def exact_cache_lookup(state: GraphState) -> dict:
    settings = get_settings()
    updates = {
        "cache_hit": False,
        "cache_type": "none",
        "llm_called": False,
        "model": settings.GROQ_MODEL,
        "exact_cache_lookup_ms": 0.0,
    }

    if not settings.ENABLE_EXACT_CACHE :
        logger.debug("Exact cache lookup skipped because it is disabled")
        return updates

    started = time.perf_counter()
    cache = ExactCache.from_settings()
    cached_answer = cache.get(state["question"])
    updates["exact_cache_lookup_ms"] = (time.perf_counter() - started) * 1000
    logger.info(
        "Exact cache lookup completed (hit=%s, duration_ms=%.2f)",
        bool(cached_answer),
        updates["exact_cache_lookup_ms"],
    )
    if cached_answer:
        updates.update(
            answer=cached_answer,
            cache_hit=True,
            cache_type="exact",
            latency_ms=updates["exact_cache_lookup_ms"],
            messages=[AIMessage(content=cached_answer)],
        )
    return updates


def semantic_cache_lookup(state: GraphState) -> dict:
    updates = {"semantic_cache_lookup_ms": 0.0}
    settings = get_settings()

    if not settings.ENABLE_SEMANTIC_CACHE :
        logger.debug("Semantic cache lookup skipped because it is disabled")
        return updates

    started = time.perf_counter()
    cache = SemanticCache.from_settings()
    match = cache.get(state["question"])
    updates["semantic_cache_lookup_ms"] = (time.perf_counter() - started) * 1000
    logger.info(
        "Semantic cache lookup completed (hit=%s, duration_ms=%.2f)",
        match is not None,
        updates["semantic_cache_lookup_ms"],
    )
    if match:
        answer, _score = match
        logger.debug("Semantic cache match selected (similarity=%.6f)", _score)
        updates.update(
            answer=answer,
            cache_hit=True,
            cache_type="semantic",
            latency_ms=updates["semantic_cache_lookup_ms"],
            messages=[AIMessage(content=answer)],
        )
    return updates


def generate_answer_node(state: GraphState) -> dict:
    settings = get_settings()
    started = time.perf_counter()
    answer = generate_answer(
        state["question"],
        conversation_messages=state["messages"],
    )
    generation_latency_ms = (time.perf_counter() - started) * 1000
    logger.info("Generated answer (model=%s, duration_ms=%.2f)", settings.GROQ_MODEL, generation_latency_ms)
    return {
        "answer": answer,
        "llm_called": True,
        "model": settings.GROQ_MODEL,
        "generation_latency_ms": generation_latency_ms,
        "latency_ms": generation_latency_ms,
        "messages": [AIMessage(content=answer)],
    }


def store_in_enabled_caches(state: GraphState) -> dict:
    settings = get_settings()

    logger.debug(
        "Storing answer in enabled caches (message_count=%d, exact_enabled=%s, semantic_enabled=%s, answer_available=%s)",
        len(state["messages"]),
        settings.ENABLE_EXACT_CACHE,
        settings.ENABLE_SEMANTIC_CACHE,
        bool(state.get("answer")),
    )

    if settings.ENABLE_EXACT_CACHE and state.get("answer"):
        try:
            success = ExactCache.from_settings().set(
                state["question"],
                state["answer"],
            )
            logger.info("Exact cache storage completed (success=%s)", success)
        except Exception:
            logger.exception("Exact cache storage failed")

    if settings.ENABLE_SEMANTIC_CACHE and state.get("answer"):
        try:
            success = SemanticCache.from_settings().set(
                state["question"],
                state["answer"],
            )
            logger.info("Semantic cache storage completed (success=%s)", success)
        except Exception:
            logger.exception("Semantic cache storage failed")

    return {}



def route_after_exact(state: GraphState) -> str:
    if state.get("cache_hit"):
        logger.debug("Routing after exact cache hit")
        return END
    logger.debug("Routing to semantic cache lookup after exact cache miss")
    return "semantic_lookup"


def route_after_semantic(state: GraphState) -> str:
    if state.get("cache_hit"):
        logger.debug("Routing after semantic cache hit")
        return END
    logger.debug("Routing to answer generation after semantic cache miss")
    return "generate"


def build_graph(checkpointer=None):
    logger.debug("Building LangGraph workflow")
    graph = StateGraph(GraphState)
    graph.add_node("exact_lookup", exact_cache_lookup)
    graph.add_node("semantic_lookup", semantic_cache_lookup)
    graph.add_node("generate", generate_answer_node)
    graph.add_node("store_caches", store_in_enabled_caches)
    graph.set_entry_point("exact_lookup")
    graph.add_conditional_edges(
        "exact_lookup",
        route_after_exact,
        {END: END, "semantic_lookup": "semantic_lookup"},
    )
    graph.add_conditional_edges(
        "semantic_lookup",
        route_after_semantic,
        {END: END, "generate": "generate"},
    )
    graph.add_edge("generate", "store_caches")
    graph.add_edge("store_caches", END)
    workflow = graph.compile(checkpointer=checkpointer)
    logger.debug("Compiled LangGraph workflow")
    return workflow


def ask_question(question: str, thread_id: str, workflow) -> GraphState:
    initial_state: GraphState = {
        "messages": [{"role": "user", "content": question}],
        "question": question,
        "answer": "",
        "cache_hit": False,
        "cache_type": "none",
        "llm_called": False,
        "model": get_settings().GROQ_MODEL,
        "latency_ms": 0.0,
    }
    result = workflow.invoke(
        initial_state,
        config={"configurable": {"thread_id": thread_id}},
    )
    logger.debug(
        "Workflow invocation completed (thread_id=%s, cache_hit=%s, cache_type=%s, llm_called=%s)",
        thread_id,
        result.get("cache_hit"),
        result.get("cache_type"),
        result.get("llm_called"),
    )
    if result.get("answer"):
        total_latency = 0.0
        for key in ("exact_cache_lookup_ms", "semantic_cache_lookup_ms", "generation_latency_ms"):
            total_latency += float(result.get(key, 0.0) or 0.0)
        result["latency_ms"] = total_latency
    return result

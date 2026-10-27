from __future__ import annotations

from typing import Annotated

try:
    from typing import NotRequired, TypedDict
except ImportError:  # Python 3.10 fallback
    from typing_extensions import NotRequired, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class GraphState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    question: str
    answer: str
    cache_hit: bool
    cache_type: str
    llm_called: bool
    model: str
    latency_ms: float
    exact_cache_lookup_ms: NotRequired[float]
    semantic_cache_lookup_ms: NotRequired[float]
    generation_latency_ms: NotRequired[float]
    cache_lookup_ms: NotRequired[float]

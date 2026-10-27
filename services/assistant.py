from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from graph import ask_question


@dataclass
class AssistantService:
    """Service facade over a graph compiled with the desired checkpointer."""

    workflow: object

    def ask(self, question: str, thread_id: str | None = None) -> dict:
        return ask_question(
            question,
            thread_id=thread_id or str(uuid4()),
            workflow=self.workflow,
        )

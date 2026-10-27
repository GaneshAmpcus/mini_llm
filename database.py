from __future__ import annotations

from datetime import datetime
from typing import Any

import psycopg
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool


class ConversationStore:
    """PostgreSQL checkpointer and conversation index."""

    def __init__(self, database_url: str) -> None:
        self.database_url = database_url
        self.pool = ConnectionPool(
            conninfo=database_url,
            min_size=1,
            max_size=10,
            kwargs={"autocommit": True},
            open=False,
        )
        self._checkpoint_connection: psycopg.Connection | None = None
        self.checkpointer: PostgresSaver | None = None

    def initialize(self) -> None:
        self.pool.open(wait=True)
        self._checkpoint_connection = psycopg.connect(
            self.database_url,
            autocommit=True,
            prepare_threshold=0,
            row_factory=dict_row,
        )
        self.checkpointer = PostgresSaver(self._checkpoint_connection)
        self.checkpointer.setup()
        with self.pool.connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    thread_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )

    def save_conversation(self, thread_id: str, title: str) -> None:
        with self.pool.connection() as connection:
            connection.execute(
                """
                INSERT INTO conversations (thread_id, title)
                VALUES (%s, %s)
                ON CONFLICT (thread_id) DO UPDATE
                SET title = EXCLUDED.title, updated_at = NOW()
                """,
                (thread_id, title),
            )

    def list_conversations(self) -> list[dict[str, Any]]:
        with self.pool.connection() as connection:
            rows = connection.execute(
                """
                SELECT thread_id, title, created_at, updated_at
                FROM conversations
                ORDER BY updated_at DESC
                """
            ).fetchall()
        return [
            {
                "thread_id": row[0],
                "title": row[1],
                "created_at": _to_isoformat(row[2]),
                "updated_at": _to_isoformat(row[3]),
            }
            for row in rows
        ]

    def get_conversation(self, thread_id: str) -> dict[str, Any] | None:
        with self.pool.connection() as connection:
            row = connection.execute(
                """
                SELECT thread_id, title, created_at, updated_at
                FROM conversations
                WHERE thread_id = %s
                """,
                (thread_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "thread_id": row[0],
            "title": row[1],
            "created_at": _to_isoformat(row[2]),
            "updated_at": _to_isoformat(row[3]),
        }

    def close(self) -> None:
        self.pool.close()
        if self._checkpoint_connection is not None:
            self._checkpoint_connection.close()


def _to_isoformat(value: datetime) -> str:
    return value.isoformat()

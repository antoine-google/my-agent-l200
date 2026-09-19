"""Persistent SQLite-backed Semantic Vector Memory Service for ADK.

Implements BaseMemoryService using Google GenAI embeddings (gemini-embedding-001)
and cosine similarity vector search over a local persistent SQLite store.
Memories persist across application restarts, process crashes, and redeployments.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sqlite3
import struct
import threading
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from google import genai
from google.adk.events.event import Event
from google.adk.memory.base_memory_service import (
    BaseMemoryService,
    SearchMemoryResponse,
)
from google.adk.memory.memory_entry import MemoryEntry
from google.adk.sessions.session import Session
from google.genai import types

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "gemini-embedding-001"
DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "vector_memory.db",
)


def _pack_vector(vector: list[float]) -> bytes:
    """Packs float list into binary bytes."""
    return struct.pack(f"{len(vector)}f", *vector)


def _unpack_vector(blob: bytes) -> list[float]:
    """Unpacks binary bytes into float list."""
    count = len(blob) // 4
    return list(struct.unpack(f"{count}f", blob))


def _cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """Computes cosine similarity between two vectors."""
    if len(v1) != len(v2) or not v1:
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return dot / (norm1 * norm2)


class PersistentVectorMemoryService(BaseMemoryService):
    """Local persistent vector memory service using SQLite and Gemini embeddings."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH) -> None:
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._lock = threading.Lock()
        self._client: genai.Client | None = None
        self._init_db()

    def _get_client(self) -> genai.Client:
        if self._client is None:
            self._client = genai.Client()
        return self._client

    def _init_db(self) -> None:
        """Initializes the SQLite schema for vector memories."""
        with self._lock, sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    app_name TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    session_id TEXT,
                    author TEXT,
                    text TEXT NOT NULL,
                    embedding BLOB,
                    timestamp TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_mem_user ON memories(app_name, user_id)"
            )
            conn.commit()

    def _embed_text(self, text: str) -> list[float] | None:
        """Generates embedding vector for a string."""
        if not text.strip():
            return None
        try:
            client = self._get_client()
            res = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=text[:2000],
            )
            if res.embeddings and res.embeddings[0].values:
                return res.embeddings[0].values
        except Exception as e:
            logger.warning("Vector embedding failed: %s. Falling back to text matching.", e)
        return None

    async def add_session_to_memory(self, session: Session) -> None:
        """Embeds and persists all meaningful dialogue turns from a session."""
        events = [
            event for event in session.events
            if event.content and event.content.parts
        ]
        await self.add_events_to_memory(
            app_name=session.app_name,
            user_id=session.user_id,
            events=events,
            session_id=session.id,
        )

    async def add_events_to_memory(
        self,
        *,
        app_name: str,
        user_id: str,
        events: Sequence[Event],
        session_id: str | None = None,
        custom_metadata: Mapping[str, object] | None = None,
    ) -> None:
        """Extracts text from event list, embeds, and persists into SQLite."""
        scoped_session_id = session_id or "default_session"

        for event in events:
            if not event.content or not event.content.parts:
                continue

            # Extract clean text from parts
            text_parts = [p.text.strip() for p in event.content.parts if getattr(p, "text", None)]
            text = " ".join(text_parts)
            if not text:
                continue

            event_id = getattr(event, "id", None) or f"{scoped_session_id}_{hash(text)}"
            author = getattr(event, "author", None) or event.content.role
            timestamp = (
                event.timestamp.isoformat()
                if hasattr(event, "timestamp") and isinstance(event.timestamp, datetime)
                else datetime.now(timezone.utc).isoformat()
            )

            # Check if already indexed
            with self._lock, sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id FROM memories WHERE id = ?", (event_id,))
                if cursor.fetchone():
                    continue

            # Generate vector embedding
            embedding = self._embed_text(text)
            embedding_blob = _pack_vector(embedding) if embedding else None

            with self._lock, sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO memories
                    (id, app_name, user_id, session_id, author, text, embedding, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (event_id, app_name, user_id, scoped_session_id, author, text, embedding_blob, timestamp),
                )
                conn.commit()

    async def search_memory(
        self,
        *,
        app_name: str,
        user_id: str,
        query: str,
    ) -> SearchMemoryResponse:
        """Executes vector cosine similarity search over stored memories."""
        if not query.strip():
            return SearchMemoryResponse(memories=[])

        query_vec = self._embed_text(query)

        with self._lock, sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, author, text, embedding, timestamp
                FROM memories
                WHERE app_name = ? AND user_id = ?
                """,
                (app_name, user_id),
            )
            rows = cursor.fetchall()

        if not rows:
            return SearchMemoryResponse(memories=[])

        scored_memories: list[tuple[float, MemoryEntry]] = []

        for row_id, author, text, embedding_blob, timestamp in rows:
            score = 0.0
            if query_vec and embedding_blob:
                doc_vec = _unpack_vector(embedding_blob)
                score = _cosine_similarity(query_vec, doc_vec)
            else:
                # Text token fallback if vector unavailable
                q_tokens = set(query.lower().split())
                doc_tokens = set(text.lower().split())
                overlap = len(q_tokens & doc_tokens)
                score = overlap / max(len(q_tokens), 1)

            entry = MemoryEntry(
                id=row_id,
                author=author,
                content=types.Content(role=author or "user", parts=[types.Part.from_text(text=text)]),
                timestamp=timestamp,
            )
            scored_memories.append((score, entry))

        # Sort descending by similarity score
        scored_memories.sort(key=lambda item: item[0], reverse=True)

        # Return top 5 relevant memories with positive similarity
        top_memories = [entry for score, entry in scored_memories[:5] if score > 0.15]

        return SearchMemoryResponse(memories=top_memories)

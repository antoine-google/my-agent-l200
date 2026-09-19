"""Unit tests for PersistentVectorMemoryService and PreloadMemory integration."""

import os
import tempfile
import pytest
from google.adk.events.event import Event
from google.genai import types

from app.app_utils.vector_memory_service import (
    PersistentVectorMemoryService,
    _cosine_similarity,
    _pack_vector,
    _unpack_vector,
)


def test_vector_packing_and_cosine():
    """Verify vector binary packing and cosine similarity calculations."""
    v1 = [1.0, 0.0, 0.0]
    v2 = [1.0, 0.0, 0.0]
    v3 = [0.0, 1.0, 0.0]

    packed = _pack_vector(v1)
    unpacked = _unpack_vector(packed)
    assert len(unpacked) == 3
    assert abs(unpacked[0] - 1.0) < 1e-5

    # Identical vectors should have similarity 1.0
    assert abs(_cosine_similarity(v1, v2) - 1.0) < 1e-5
    # Orthogonal vectors should have similarity 0.0
    assert abs(_cosine_similarity(v1, v3) - 0.0) < 1e-5


@pytest.mark.asyncio
async def test_persistent_memory_service_storage_and_search():
    """Verify storing dialogue events into persistent SQLite and querying them."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_memory.db")
        service = PersistentVectorMemoryService(db_path=db_path)

        event = Event(
            id="evt_test_1",
            author="user",
            content=types.Content(
                role="user",
                parts=[types.Part.from_text(text="I prefer to track GOOGL 50-day moving average closely.")],
            ),
        )

        await service.add_events_to_memory(
            app_name="app",
            user_id="trader_1",
            events=[event],
        )

        # Query memory
        res = await service.search_memory(
            app_name="app",
            user_id="trader_1",
            query="What moving average does the user track?",
        )

        assert len(res.memories) >= 1
        found_text = res.memories[0].content.parts[0].text
        assert "GOOGL 50-day moving average" in found_text

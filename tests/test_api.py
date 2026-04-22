"""Tests for src.ui.api — FastAPI + SSE server (Unit 7 — Phase 3)."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.core.config import load_config
from src.core.types import StreamChunk, ToolCall
from src.ui.api import _event_to_sse, _sse_frame, create_app, SessionRegistry


# ---------------------------------------------------------------------------
# Session registry
# ---------------------------------------------------------------------------


def test_session_registry_create_and_get():
    reg = SessionRegistry()
    s = reg.create()
    assert s.id
    assert s.active is True
    assert reg.get(s.id) is s
    assert reg.active_count() == 1
    assert reg.total_count() == 1


def test_session_registry_custom_id():
    reg = SessionRegistry()
    s = reg.create(session_id="my-id")
    assert s.id == "my-id"


def test_session_registry_mark_complete():
    reg = SessionRegistry()
    s = reg.create()
    reg.mark_complete(s.id)
    assert reg.active_count() == 0
    assert reg.total_count() == 1  # still retained for /trajectory


def test_session_registry_missing_session_returns_none():
    reg = SessionRegistry()
    assert reg.get("nope") is None


# ---------------------------------------------------------------------------
# SSE framing
# ---------------------------------------------------------------------------


def test_sse_frame_format():
    frame = _sse_frame("TokenEvent", {"token": "hi"})
    assert frame.startswith("event: TokenEvent\n")
    assert "data: " in frame
    assert frame.endswith("\n\n")


def test_event_to_sse_with_dataclass():
    from src.core.types import TokenEvent
    frame = _event_to_sse(TokenEvent(token="x"))
    assert "event: TokenEvent" in frame
    assert '"token": "x"' in frame


def test_event_to_sse_non_dataclass_falls_back_to_repr():
    class NotADataclass:
        def __repr__(self):
            return "custom repr"
    frame = _event_to_sse(NotADataclass())
    assert "custom repr" in frame


# ---------------------------------------------------------------------------
# HTTP endpoints — TestClient
# ---------------------------------------------------------------------------


def _mock_provider(text: str):
    provider = MagicMock()

    async def stream_message(**kwargs) -> AsyncGenerator:
        for word in text.split():
            yield StreamChunk(type="token", data=word + " ")
        yield StreamChunk(type="done", data="stop")

    provider.stream_message = stream_message
    return provider


def test_status_endpoint_returns_counts():
    app = create_app(config=load_config())
    client = TestClient(app)
    response = client.get("/status")
    assert response.status_code == 200
    body = response.json()
    assert body["active_sessions"] == 0
    assert "budget_default" in body


def test_trajectory_missing_session_404s():
    app = create_app(config=load_config())
    client = TestClient(app)
    response = client.get("/trajectory/nonexistent")
    assert response.status_code == 404


def test_chat_streams_sse_frames():
    config = load_config()
    config.provider.auxiliary = None

    with patch("src.ui.api.resolve_provider", return_value=_mock_provider("hi there")), \
         patch("src.ui.api.resolve_auxiliary", return_value=None):
        app = create_app(config=config)
        client = TestClient(app)
        with client.stream("POST", "/chat", json={
            "messages": [{"role": "user", "content": "hello"}],
        }) as response:
            assert response.status_code == 200
            assert "text/event-stream" in response.headers["content-type"]
            # Accumulate all SSE frames
            body = b"".join(response.iter_bytes()).decode("utf-8")

    assert "event: FinalAnswerEvent" in body or "event: TokenEvent" in body


def test_chat_includes_session_id_header():
    config = load_config()
    config.provider.auxiliary = None

    with patch("src.ui.api.resolve_provider", return_value=_mock_provider("ok")), \
         patch("src.ui.api.resolve_auxiliary", return_value=None):
        app = create_app(config=config)
        client = TestClient(app)
        with client.stream("POST", "/chat", json={
            "messages": [{"role": "user", "content": "hi"}],
            "session_id": "custom-abc",
        }) as response:
            body = b"".join(response.iter_bytes())

        # Session should be registered after stream completes
        assert app.state.sessions.get("custom-abc") is not None
        # /trajectory should now return data for that session
        response = client.get("/trajectory/custom-abc")
        assert response.status_code == 200
        payload = response.json()
        assert payload["id"] == "custom-abc"
        assert "conversations" in payload


def test_chat_emits_error_event_on_provider_exception():
    config = load_config()
    config.provider.auxiliary = None
    config.provider.fallback = None  # force error surface

    failing = MagicMock()

    async def stream_message(**kwargs):
        raise RuntimeError("upstream down")
        yield  # pragma: no cover

    failing.stream_message = stream_message

    with patch("src.ui.api.resolve_provider", return_value=failing), \
         patch("src.ui.api.resolve_auxiliary", return_value=None):
        app = create_app(config=config)
        client = TestClient(app)
        with client.stream("POST", "/chat", json={
            "messages": [{"role": "user", "content": "hi"}],
        }) as response:
            body = b"".join(response.iter_bytes()).decode("utf-8")

    # Either the agent-loop's ErrorEvent or the api's outer fallback
    assert "event: ErrorEvent" in body

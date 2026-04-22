"""Tests for NudgeEngine (Unit 2 — Phase 3)."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.core.types import Message, StreamChunk, ToolCall, ToolResult
from src.learning.nudge import NudgeEngine, NudgeResult, _parse_response
from src.learning.trajectory import TrajectoryRecorder
from src.memory.markdown_store import MarkdownMemoryStore


# ---------------------------------------------------------------------------
# Response parsing — pure function, no I/O
# ---------------------------------------------------------------------------


def test_parse_null_response():
    assert _parse_response("null") is None
    assert _parse_response("  null  ") is None
    assert _parse_response("") is None


def test_parse_valid_json():
    text = '{"topic": "user-prefs", "hook": "likes terse answers", "body": "User prefers one-line replies."}'
    result = _parse_response(text)
    assert result == ("user-prefs", "likes terse answers", "User prefers one-line replies.")


def test_parse_code_fenced_json():
    text = '```json\n{"topic": "x", "hook": "y", "body": "z"}\n```'
    assert _parse_response(text) == ("x", "y", "z")


def test_parse_missing_field_returns_none():
    text = '{"topic": "x", "hook": "y"}'  # no body
    assert _parse_response(text) is None


def test_parse_blank_field_returns_none():
    text = '{"topic": "x", "hook": "y", "body": "   "}'
    assert _parse_response(text) is None


def test_parse_garbage_returns_none():
    assert _parse_response("this is not json") is None


def test_parse_json_embedded_in_prose():
    text = 'Thinking... here it is: {"topic": "a", "hook": "b", "body": "c"} done.'
    assert _parse_response(text) == ("a", "b", "c")


# ---------------------------------------------------------------------------
# NudgeEngine — with mocked aux provider and real MarkdownMemoryStore
# ---------------------------------------------------------------------------


def _mock_provider(response_text: str):
    """Provider that streams one token then done, returning response_text."""
    provider = MagicMock()

    async def stream_message(**kwargs) -> AsyncGenerator:
        yield StreamChunk(type="token", data=response_text)
        yield StreamChunk(type="done", data="stop")

    provider.stream_message = stream_message
    return provider


def _failing_provider():
    provider = MagicMock()

    async def stream_message(**kwargs) -> AsyncGenerator:
        raise RuntimeError("upstream blew up")
        yield  # pragma: no cover — make this an async generator

    provider.stream_message = stream_message
    return provider


def _recorder_with_turns(n: int = 3) -> TrajectoryRecorder:
    rec = TrajectoryRecorder(session_id="s")
    for i in range(n):
        rec.record_turn(
            user_msg=Message(role="user", content=f"ask {i}") if i == 0 else None,
            assistant_msg=Message(role="assistant", content=None),
            tool_calls=[ToolCall(id=f"c{i}", name="greet", arguments={})],
            tool_results=[ToolResult(tool_call_id=f"c{i}", name="greet", content="ok")],
        )
    return rec


async def test_nudge_writes_memory_on_valid_response(tmp_path: Path):
    store = MarkdownMemoryStore(root=tmp_path)
    provider = _mock_provider(
        '{"topic": "user-terse", "hook": "likes terse answers", '
        '"body": "User prefers short replies."}'
    )
    engine = NudgeEngine(provider, store, interval=15)

    rec = _recorder_with_turns(3)
    result = await engine.maybe_nudge(rec.turns, session_id="s1")

    assert result.saved is True
    assert result.topic == "user-terse"
    assert result.reason == "wrote"
    # File on disk
    topic_file = tmp_path / "user-terse.md"
    assert topic_file.exists()
    assert "short replies" in topic_file.read_text(encoding="utf-8")
    # Index entry
    index = (tmp_path / "MEMORY.md").read_text(encoding="utf-8")
    assert "likes terse answers" in index


async def test_nudge_skips_on_null_response(tmp_path: Path):
    store = MarkdownMemoryStore(root=tmp_path)
    provider = _mock_provider("null")
    engine = NudgeEngine(provider, store, interval=15)

    result = await engine.maybe_nudge(_recorder_with_turns().turns, session_id="s1")

    assert result.saved is False
    assert result.reason == "skipped"
    # No files written
    assert not (tmp_path / "MEMORY.md").exists()


async def test_nudge_skips_on_empty_turns(tmp_path: Path):
    store = MarkdownMemoryStore(root=tmp_path)
    provider = _mock_provider("null")
    engine = NudgeEngine(provider, store, interval=15)

    result = await engine.maybe_nudge([], session_id="s1")

    assert result.saved is False
    assert result.reason == "no_turns"


async def test_nudge_circuit_breaker_trips_after_3_failures(tmp_path: Path):
    store = MarkdownMemoryStore(root=tmp_path)
    provider = _failing_provider()
    engine = NudgeEngine(provider, store, interval=15, breaker_threshold=3)

    turns = _recorder_with_turns().turns
    for _ in range(3):
        r = await engine.maybe_nudge(turns, session_id="s1")
        assert r.reason == "aux_error"
    # 4th call: breaker open, skip without calling provider
    r4 = await engine.maybe_nudge(turns, session_id="s1")
    assert r4.reason == "breaker_open"
    assert engine.is_available is False


async def test_nudge_breaker_recovers_on_success(tmp_path: Path):
    store = MarkdownMemoryStore(root=tmp_path)

    call_count = {"n": 0}
    provider = MagicMock()

    async def stream_message(**kwargs) -> AsyncGenerator:
        call_count["n"] += 1
        if call_count["n"] <= 2:
            raise RuntimeError("flaky")
        yield StreamChunk(type="token", data='{"topic": "t", "hook": "h", "body": "b"}')
        yield StreamChunk(type="done", data="stop")

    provider.stream_message = stream_message
    engine = NudgeEngine(provider, store, interval=15, breaker_threshold=3)

    # Two failures, then a success — breaker should have reset
    turns = _recorder_with_turns().turns
    await engine.maybe_nudge(turns, session_id="s1")
    await engine.maybe_nudge(turns, session_id="s1")
    assert engine.is_available is True  # still below threshold
    r = await engine.maybe_nudge(turns, session_id="s1")
    assert r.saved is True
    assert engine.is_available is True


async def test_nudge_handles_garbage_response(tmp_path: Path):
    store = MarkdownMemoryStore(root=tmp_path)
    provider = _mock_provider("??? not json ???")
    engine = NudgeEngine(provider, store, interval=15)

    result = await engine.maybe_nudge(_recorder_with_turns().turns, session_id="s1")

    assert result.saved is False
    assert result.reason == "skipped"


async def test_nudge_no_provider_gracefully_skips(tmp_path: Path):
    store = MarkdownMemoryStore(root=tmp_path)
    engine = NudgeEngine(None, store, interval=15)

    result = await engine.maybe_nudge(_recorder_with_turns().turns, session_id="s1")

    assert result.saved is False
    assert result.reason == "no_provider"
    assert engine.is_available is False


# ---------------------------------------------------------------------------
# Integration: agent_loop triggers nudge at the right interval
# ---------------------------------------------------------------------------


async def test_agent_loop_triggers_nudge_at_interval(tmp_path: Path):
    """Agent with 3 tool-call turns, nudge_interval=2 → nudge fires once
    after turn 2 (bucket 0→1), once more after turn 3 (bucket 1→... wait,
    3//2=1, so only 1 bucket crossing over 3 tool calls). Actually 3//2=1
    so still bucket 1. Let's use 4 turns and interval=2 → buckets 1 and 2.
    """
    from unittest.mock import AsyncMock

    from src.core.agent_loop import agent_loop
    from src.core.config import load_config
    from src.core.env_injector import EnvironmentInjector
    from src.core.iteration_budget import IterationBudget
    from src.core.plan_mode import PlanMode
    from src.core.streaming_executor import StreamingExecutor
    from src.core.types import (
        PermissionLevel, ToolContext, ToolDefinition, ToolSchema,
    )
    from src.tools.registry import ToolRegistry

    reg = ToolRegistry()

    async def noop(**kw) -> str:
        return "ok"

    reg.register(ToolDefinition(
        name="noop",
        description="noop",
        handler=noop,
        schema=ToolSchema(
            name="noop", description="noop",
            parameters={"type": "object", "properties": {}},
            permission_level=PermissionLevel.READ_ONLY,
        ),
    ))

    config = load_config()
    config.agent.nudge_interval = 2
    pm = PlanMode()
    ctx = ToolContext(plan_mode=pm, registry=reg, config=config, working_dir=tmp_path)
    executor = StreamingExecutor(reg, ctx)

    # Provider: 4 tool-call turns then a final answer
    call_n = {"n": 0}
    provider = MagicMock()

    async def stream_message(**kwargs) -> AsyncGenerator:
        call_n["n"] += 1
        if call_n["n"] <= 4:
            yield StreamChunk(type="tool_call_start", data={"id": f"c{call_n['n']}", "name": "noop"})
            yield StreamChunk(
                type="tool_call_complete",
                data=ToolCall(id=f"c{call_n['n']}", name="noop", arguments={}),
            )
            yield StreamChunk(type="done", data="tool_calls")
        else:
            yield StreamChunk(type="token", data="done")
            yield StreamChunk(type="done", data="stop")

    provider.stream_message = stream_message

    # Learning container with spy nudge
    recorder = TrajectoryRecorder(session_id="int")
    nudge_calls: list[int] = []

    @dataclass
    class _Learning:
        recorder: TrajectoryRecorder
        nudge: object
        session_id: str

    class _SpyNudge:
        async def maybe_nudge(self, turns, session_id):
            nudge_calls.append(len(turns))
            return NudgeResult(saved=False, reason="skipped")

    learning = _Learning(recorder=recorder, nudge=_SpyNudge(), session_id="int")

    gen = agent_loop(
        conversation=[Message(role="user", content="go")],
        tools=reg,
        budget=IterationBudget(20),
        provider=provider,
        executor=executor,
        env_injector=EnvironmentInjector(),
        plan_mode=pm,
        config=config,
        trajectory=recorder,
        learning=learning,
    )
    async for _ in gen:
        pass

    # 4 tool calls, interval=2 → buckets 1 and 2 → 2 nudges fired.
    assert len(nudge_calls) == 2, f"expected 2 nudges, got {len(nudge_calls)}"

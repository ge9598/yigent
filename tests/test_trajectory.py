"""Tests for TrajectoryRecorder (Unit 1 — Phase 3)."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.core.agent_loop import agent_loop
from src.core.config import load_config
from src.core.env_injector import EnvironmentInjector
from src.core.iteration_budget import IterationBudget
from src.core.plan_mode import PlanMode
from src.core.streaming_executor import StreamingExecutor
from src.core.types import (
    Message, PermissionLevel, StreamChunk, ToolCall, ToolContext,
    ToolDefinition, ToolResult, ToolSchema,
)
from src.learning.trajectory import TrajectoryRecorder, TurnRecord
from src.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Unit-level tests: TrajectoryRecorder as a standalone component
# ---------------------------------------------------------------------------


def test_record_turn_basic():
    rec = TrajectoryRecorder(session_id="s1")
    user = Message(role="user", content="hi")
    assistant = Message(role="assistant", content="hello")

    record = rec.record_turn(user_msg=user, assistant_msg=assistant)

    assert isinstance(record, TurnRecord)
    assert record.turn_index == 0
    assert record.user_msg == user
    assert record.assistant_msg == assistant
    assert record.tool_calls == []
    assert record.tool_results == []
    assert len(rec) == 1


def test_turn_indices_increment():
    rec = TrajectoryRecorder(session_id="s1")
    for i in range(3):
        rec.record_turn(assistant_msg=Message(role="assistant", content=f"r{i}"))
    assert [t.turn_index for t in rec.turns] == [0, 1, 2]


def test_attach_tool_results_updates_last_turn():
    rec = TrajectoryRecorder(session_id="s1")
    tc = ToolCall(id="c1", name="greet", arguments={"name": "world"})
    rec.record_turn(
        assistant_msg=Message(role="assistant", content=None),
        tool_calls=[tc],
    )
    result = ToolResult(tool_call_id="c1", name="greet", content="Hello, world!")
    rec.attach_tool_results([result])

    assert rec.turns[0].tool_results == [result]


def test_attach_tool_results_on_empty_is_noop():
    rec = TrajectoryRecorder(session_id="s1")
    rec.attach_tool_results([ToolResult(tool_call_id="x", name="y", content="z")])
    assert len(rec) == 0  # no crash, no ghost turn


def test_export_sharegpt_shape():
    rec = TrajectoryRecorder(session_id="session-42")
    rec.record_turn(
        user_msg=Message(role="user", content="greet me"),
        assistant_msg=Message(role="assistant", content=None),
        tool_calls=[ToolCall(id="c1", name="greet", arguments={"name": "葛"})],
        tool_results=[ToolResult(tool_call_id="c1", name="greet", content="Hello, 葛!")],
    )
    rec.record_turn(
        assistant_msg=Message(role="assistant", content="Done."),
    )

    out = rec.export_sharegpt()

    assert out["id"] == "session-42"
    convs = out["conversations"]
    # human → gpt(with tool_calls) → tool → gpt(final)
    assert [c["from"] for c in convs] == ["human", "gpt", "tool", "gpt"]
    assert convs[1]["tool_calls"][0]["name"] == "greet"
    assert convs[1]["tool_calls"][0]["arguments"] == {"name": "葛"}
    assert convs[2]["value"] == "Hello, 葛!"
    assert convs[2]["is_error"] is False
    assert convs[3]["value"] == "Done."


def test_export_rl_terminal_and_nonterminal():
    rec = TrajectoryRecorder(session_id="s1")
    rec.record_turn(
        user_msg=Message(role="user", content="go"),
        assistant_msg=Message(role="assistant", content=None),
        tool_calls=[ToolCall(id="c1", name="t", arguments={})],
        tool_results=[ToolResult(tool_call_id="c1", name="t", content="ok")],
    )
    rec.record_turn(
        assistant_msg=Message(role="assistant", content="done"),
    )

    transitions = rec.export_rl()

    assert len(transitions) == 2
    # First: tool transition, not terminal
    assert transitions[0]["action"]["tool_calls"][0]["name"] == "t"
    assert transitions[0]["next_state"] is not None
    assert transitions[0]["reward"] is None
    assert transitions[0].get("terminal") is not True
    # Second: final-answer, terminal
    assert transitions[1]["terminal"] is True
    assert transitions[1]["next_state"] is None
    assert transitions[1]["action"]["final_answer"] == "done"


def test_save_sharegpt_creates_file(tmp_path: Path):
    rec = TrajectoryRecorder(session_id="s1")
    rec.record_turn(
        user_msg=Message(role="user", content="hi"),
        assistant_msg=Message(role="assistant", content="hello"),
    )
    out = rec.save(tmp_path / "sub" / "traj.json", fmt="sharegpt")

    assert out.exists()
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["id"] == "s1"
    assert loaded["conversations"][0]["value"] == "hi"


def test_save_rejects_unknown_format(tmp_path: Path):
    rec = TrajectoryRecorder(session_id="s1")
    rec.record_turn(assistant_msg=Message(role="assistant", content="ok"))
    with pytest.raises(ValueError, match="Unknown format"):
        rec.save(tmp_path / "x.json", fmt="bogus")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Integration test: recorder wired through agent_loop records real turns
# ---------------------------------------------------------------------------


def _tool_then_text_provider(tool_name: str, args: dict, answer: str):
    provider = MagicMock()
    call_count = {"n": 0}

    async def stream_message(**kwargs) -> AsyncGenerator:
        call_count["n"] += 1
        if call_count["n"] == 1:
            yield StreamChunk(type="tool_call_start", data={"id": "c1", "name": tool_name})
            yield StreamChunk(
                type="tool_call_complete",
                data=ToolCall(id="c1", name=tool_name, arguments=args),
            )
            yield StreamChunk(type="done", data="tool_calls")
        else:
            for w in answer.split():
                yield StreamChunk(type="token", data=w + " ")
            yield StreamChunk(type="done", data="stop")

    provider.stream_message = stream_message
    return provider


def _make_registry() -> ToolRegistry:
    reg = ToolRegistry()

    async def greet(name: str) -> str:
        return f"Hello, {name}!"

    reg.register(ToolDefinition(
        name="greet",
        description="Greet someone",
        handler=greet,
        schema=ToolSchema(
            name="greet",
            description="Greet someone",
            parameters={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
            permission_level=PermissionLevel.READ_ONLY,
        ),
    ))
    return reg


async def test_agent_loop_records_user_tool_and_final():
    reg = _make_registry()
    config = load_config()
    pm = PlanMode()
    ctx = ToolContext(plan_mode=pm, registry=reg, config=config, working_dir=Path.cwd())
    executor = StreamingExecutor(reg, ctx)
    recorder = TrajectoryRecorder(session_id="int-1")

    conversation: list[Message] = [Message(role="user", content="greet 葛")]
    gen = agent_loop(
        conversation=conversation,
        tools=reg,
        budget=IterationBudget(10),
        provider=_tool_then_text_provider("greet", {"name": "葛"}, "Done."),
        executor=executor,
        env_injector=EnvironmentInjector(),
        plan_mode=pm,
        config=config,
        trajectory=recorder,
    )
    async for _ in gen:
        pass

    # Expect 2 recorded turns: the tool-call turn (with user attached) and
    # the final-answer turn (no user, no tool calls).
    turns = recorder.turns
    assert len(turns) == 2

    t0 = turns[0]
    assert t0.user_msg is not None
    assert t0.user_msg["content"] == "greet 葛"
    assert len(t0.tool_calls) == 1
    assert t0.tool_calls[0].name == "greet"
    assert len(t0.tool_results) == 1
    assert "Hello, 葛!" in t0.tool_results[0].content

    t1 = turns[1]
    assert t1.user_msg is None           # same user msg, attached to turn 0 only
    assert t1.tool_calls == []
    assert "Done." in (t1.assistant_msg.get("content") or "")


async def test_agent_loop_without_trajectory_is_unaffected():
    """Sanity: if trajectory=None, agent_loop behaves identically."""
    reg = _make_registry()
    config = load_config()
    pm = PlanMode()
    ctx = ToolContext(plan_mode=pm, registry=reg, config=config, working_dir=Path.cwd())
    executor = StreamingExecutor(reg, ctx)

    conversation: list[Message] = [Message(role="user", content="hi")]

    provider = MagicMock()

    async def stream_message(**kwargs) -> AsyncGenerator:
        yield StreamChunk(type="token", data="hello")
        yield StreamChunk(type="done", data="stop")

    provider.stream_message = stream_message

    gen = agent_loop(
        conversation=conversation,
        tools=reg,
        budget=IterationBudget(10),
        provider=provider,
        executor=executor,
        env_injector=EnvironmentInjector(),
        plan_mode=pm,
        config=config,
        trajectory=None,
    )
    events = [ev async for ev in gen]
    assert any(type(ev).__name__ == "FinalAnswerEvent" for ev in events)

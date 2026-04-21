"""Regression tests for Fix 7 — agent_loop must clean up dispatched tool tasks
on every exit path.

Before the fix, ``pending_dispatched`` was only cleared on the
"primary-failed-will-try-fallback" path. Four paths leaked:
  - primary failed, no fallback configured
  - primary failed, fallback build failed
  - fallback stream also failed
  - Ctrl+C / CancelledError

We verify the no-fallback path here (easiest to construct deterministically).
"""

from __future__ import annotations

import asyncio
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
    ErrorEvent, Message, PermissionLevel, StreamChunk, ToolCall,
    ToolContext, ToolDefinition, ToolSchema,
)
from src.tools.registry import ToolRegistry


def _slow_tool_registry(started: asyncio.Event) -> ToolRegistry:
    """Registry with a single tool that blocks until cancelled."""
    reg = ToolRegistry()

    async def blocks_forever() -> str:
        started.set()
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            # Cooperative cancellation acknowledged.
            raise
        return "never"  # pragma: no cover

    reg.register(ToolDefinition(
        name="blocks_forever",
        description="Hangs until cancelled — for cleanup tests.",
        handler=blocks_forever,
        schema=ToolSchema(
            name="blocks_forever",
            description="Hangs until cancelled.",
            parameters={"type": "object", "properties": {}, "required": []},
            permission_level=PermissionLevel.READ_ONLY,
        ),
    ))
    return reg


def _dispatch_then_fail_provider(tool_name: str):
    """Provider that emits a tool_call_complete then raises mid-stream.

    This simulates: tool was dispatched and started running, then the
    primary provider failed before the stream finished. The agent_loop
    must clean up the pending task.
    """
    provider = MagicMock()

    async def stream_message(**kwargs) -> AsyncGenerator:
        yield StreamChunk(
            type="tool_call_start",
            data={"id": "c1", "name": tool_name},
        )
        yield StreamChunk(
            type="tool_call_complete",
            data=ToolCall(id="c1", name=tool_name, arguments={}),
        )
        # Yield a brief pause so the dispatched task actually starts.
        await asyncio.sleep(0.01)
        raise RuntimeError("provider mid-stream failure")

    provider.stream_message = stream_message
    return provider


@pytest.mark.asyncio
async def test_pending_dispatched_tasks_cleaned_on_provider_failure_no_fallback():
    """Primary fails, no fallback configured → pending tool tasks are cancelled."""
    started = asyncio.Event()
    reg = _slow_tool_registry(started)
    config = load_config()
    # Explicitly nil the fallback so no retry happens.
    config.provider.fallback = None
    pm = PlanMode()
    ctx = ToolContext(plan_mode=pm, registry=reg, config=config, working_dir=Path.cwd())
    executor = StreamingExecutor(reg, ctx)

    conversation = [Message(role="user", content="hang please")]

    tasks_before = {t for t in asyncio.all_tasks() if not t.done()}

    events = []
    async for event in agent_loop(
        conversation=conversation,
        tools=reg,
        budget=IterationBudget(10),
        provider=_dispatch_then_fail_provider("blocks_forever"),
        executor=executor,
        env_injector=EnvironmentInjector(),
        plan_mode=pm,
        config=config,
    ):
        events.append(event)

    # The loop surfaced an error event (not silently swallowed).
    assert any(isinstance(e, ErrorEvent) and not e.recoverable for e in events), \
        "expected non-recoverable ErrorEvent after provider failure"

    # Give the event loop one tick to finish reaping.
    await asyncio.sleep(0.05)

    tasks_after = {t for t in asyncio.all_tasks() if not t.done()}
    new_leaked = tasks_after - tasks_before
    # No dispatched tool task should survive — the 'blocks_forever' task
    # would be the leak. Any other task in `new_leaked` is unrelated
    # infrastructure noise (pytest-asyncio, anyio, etc.). We assert that
    # nothing in the leak set has "blocks_forever" in its coro name.
    for t in new_leaked:
        coro = getattr(t, "get_coro", lambda: None)()
        name = getattr(coro, "__qualname__", "") or getattr(coro, "__name__", "") or ""
        assert "blocks_forever" not in name, (
            f"leaked tool task survived: {t!r} coro={name!r}"
        )

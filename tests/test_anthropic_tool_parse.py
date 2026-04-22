"""Tests for anthropic_compat tool_use input parse failure handling.

Regression: MiniMax's /anthropic gateway has been observed to drop runs
of whitespace mid-stream inside input_json_delta fragments, producing
an accumulated buffer that is no longer valid JSON. Previously this
surfaced as `missing 2 required positional arguments` when the handler
was called with an empty dict after _filter_kwargs stripped the
unrecognized `_raw` key. The fix:

- Parse failure attaches a `__parse_error__` sentinel with structured
  failure info (offset, buffer_len, delta_count, msg).
- StreamingExecutor._execute_single detects the sentinel and returns
  an actionable tool_result asking the model to retry.
- AnthropicCompatProvider logs the raw delta sequence + hex window
  around the parse offset for post-mortem analysis.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.core.config import load_config
from src.core.streaming_executor import StreamingExecutor
from src.core.types import (
    PermissionDecision, PermissionLevel, ToolCall, ToolContext,
    ToolDefinition, ToolSchema,
)
from src.providers.anthropic_compat import (
    AnthropicCompatProvider, _ToolUseAccumulator,
)
from src.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# _parse_accumulated — direct unit tests on the accumulator
# ---------------------------------------------------------------------------


def test_parse_accumulated_valid_json():
    acc = _ToolUseAccumulator(index=0, id="c1", name="greet")
    acc.arguments_buffer = '{"name": "world"}'
    acc.delta_history = ['{"name": "', 'world"}']
    tc = AnthropicCompatProvider._parse_accumulated(acc)
    assert tc.arguments == {"name": "world"}
    assert "__parse_error__" not in tc.arguments


def test_parse_accumulated_empty_buffer():
    acc = _ToolUseAccumulator(index=0, id="c1", name="no_args_tool")
    tc = AnthropicCompatProvider._parse_accumulated(acc)
    assert tc.arguments == {}


def test_parse_accumulated_malformed_attaches_sentinel():
    acc = _ToolUseAccumulator(index=0, id="c1", name="write_file")
    # Missing closing brace — realistic shape of the MiniMax corruption
    acc.arguments_buffer = '{"path": "a.py", "content": "x'
    acc.delta_history = [acc.arguments_buffer]
    tc = AnthropicCompatProvider._parse_accumulated(acc)
    assert "__parse_error__" in tc.arguments
    err = tc.arguments["__parse_error__"]
    assert err["buffer_len"] == len(acc.arguments_buffer)
    assert err["delta_count"] == 1
    assert isinstance(err["offset"], int)
    assert "msg" in err


def test_parse_accumulated_logs_forensic_data(caplog):
    acc = _ToolUseAccumulator(index=0, id="call_x", name="write_file")
    # Simulate the MiniMax whitespace-drop pattern: third delta missing
    # a whole run of spaces, producing unbalanced JSON.
    acc.delta_history = [
        '{"path": "src/a.py", "content": "',
        'def foo():\\n    x = 1\\n',
        '}',  # closes prematurely, no closing quote on content
    ]
    acc.arguments_buffer = "".join(acc.delta_history)

    with caplog.at_level(logging.WARNING, logger="src.providers.anthropic_compat"):
        AnthropicCompatProvider._parse_accumulated(acc)

    messages = [r.getMessage() for r in caplog.records]
    combined = "\n".join(messages)
    # Structured pieces that should always appear:
    assert "Failed to parse tool_use input for write_file" in combined
    assert "id=call_x" in combined
    assert "delta_lens=" in combined
    assert "hex:" in combined
    # The last few deltas are logged verbatim
    assert any("last delta[" in m for m in messages)


# ---------------------------------------------------------------------------
# StreamingExecutor — sentinel triggers actionable error tool_result
# ---------------------------------------------------------------------------


def _make_registry_with_write():
    reg = ToolRegistry()

    async def _write(path: str, content: str) -> str:
        Path(path).write_text(content, encoding="utf-8")
        return f"wrote {len(content)} bytes"

    reg.register(ToolDefinition(
        name="write_file",
        description="Write content to a file",
        handler=_write,
        schema=ToolSchema(
            name="write_file",
            description="Write content to a file",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
            permission_level=PermissionLevel.WRITE,
        ),
    ))
    return reg


async def test_executor_sentinel_produces_actionable_error(tmp_path):
    reg = _make_registry_with_write()
    ctx = ToolContext(
        plan_mode=MagicMock(is_active=False),
        registry=reg,
        config=load_config(),
        working_dir=tmp_path,
    )
    executor = StreamingExecutor(reg, ctx)

    tc = ToolCall(
        id="c1", name="write_file",
        arguments={
            "__parse_error__": {
                "reason": "tool_use input JSON failed to parse",
                "msg": "Unterminated string starting at",
                "offset": 1523,
                "buffer_len": 2047,
                "delta_count": 9,
            },
        },
    )

    result = await executor._execute_single(tc)

    assert result.is_error is True
    assert "could not be parsed as JSON" in result.content
    assert "offset 1523" in result.content
    assert "2047 bytes" in result.content
    assert "9 deltas" in result.content
    # Actionable retry advice
    assert "retry" in result.content.lower()


async def test_executor_does_not_call_handler_on_parse_error(tmp_path):
    """Regression: the broken path was _filter_kwargs dropping _raw and
    calling write_file() with zero args. The sentinel path must short-
    circuit before handler lookup."""
    reg = ToolRegistry()
    called = {"n": 0}

    async def _should_never_run(path: str, content: str) -> str:
        called["n"] += 1
        return "nope"

    reg.register(ToolDefinition(
        name="write_file",
        description="Write content to a file",
        handler=_should_never_run,
        schema=ToolSchema(
            name="write_file",
            description="Write content to a file",
            parameters={"type": "object", "properties": {}},
            permission_level=PermissionLevel.WRITE,
        ),
    ))

    ctx = ToolContext(
        plan_mode=MagicMock(is_active=False),
        registry=reg,
        config=load_config(),
        working_dir=tmp_path,
    )
    executor = StreamingExecutor(reg, ctx)

    tc = ToolCall(
        id="c1", name="write_file",
        arguments={"__parse_error__": {"msg": "x", "offset": 0,
                                        "buffer_len": 0, "delta_count": 0}},
    )
    result = await executor._execute_single(tc)

    assert called["n"] == 0
    assert result.is_error is True


async def test_executor_ignores_non_dict_sentinel(tmp_path):
    """A key named __parse_error__ with a non-dict value must not trigger
    the error path — the sentinel is specifically a dict with structured
    fields."""
    reg = _make_registry_with_write()
    ctx = ToolContext(
        plan_mode=MagicMock(is_active=False),
        registry=reg,
        config=load_config(),
        working_dir=tmp_path,
    )
    executor = StreamingExecutor(reg, ctx)

    # Model legitimately passes "__parse_error__" as a string somewhere — the
    # executor should treat this as a normal arg (filtered out by signature)
    # rather than triggering the parse-error path.
    tc = ToolCall(
        id="c1", name="write_file",
        arguments={
            "__parse_error__": "not a dict — real call",
            "path": str(tmp_path / "a.txt"),
            "content": "hi",
        },
    )
    result = await executor._execute_single(tc)

    assert result.is_error is False
    assert "wrote 2 bytes" in result.content

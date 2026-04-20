"""Tests for AnthropicCompatProvider — message translation + stream parsing."""

from __future__ import annotations

import pytest

from src.core.types import Message, PermissionLevel, ToolSchema
from src.providers.anthropic_compat import AnthropicCompatProvider


# ---------------------------------------------------------------------------
# Message translation
# ---------------------------------------------------------------------------

def test_translate_extracts_system_and_drops_from_messages() -> None:
    messages: list[Message] = [
        {"role": "system", "content": "you are helpful"},
        {"role": "user", "content": "hi"},
    ]
    system, out = AnthropicCompatProvider._translate_messages(messages)
    assert system == "you are helpful"
    assert out == [{"role": "user", "content": "hi"}]


def test_translate_concatenates_multiple_system_messages() -> None:
    messages: list[Message] = [
        {"role": "system", "content": "rule A"},
        {"role": "system", "content": "rule B"},
        {"role": "user", "content": "hi"},
    ]
    system, out = AnthropicCompatProvider._translate_messages(messages)
    assert system == "rule A\n\nrule B"
    assert len(out) == 1


def test_translate_assistant_tool_calls_become_tool_use_blocks() -> None:
    messages: list[Message] = [
        {"role": "user", "content": "read foo"},
        {
            "role": "assistant",
            "content": "reading",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path": "foo"}'},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "file body",
        },
    ]
    _, out = AnthropicCompatProvider._translate_messages(messages)
    assert len(out) == 3
    # user
    assert out[0] == {"role": "user", "content": "read foo"}
    # assistant with two blocks: text + tool_use
    assistant = out[1]
    assert assistant["role"] == "assistant"
    blocks = assistant["content"]
    assert blocks[0] == {"type": "text", "text": "reading"}
    assert blocks[1] == {
        "type": "tool_use",
        "id": "call_1",
        "name": "read_file",
        "input": {"path": "foo"},
    }
    # tool result flushed as user message
    assert out[2] == {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "call_1", "content": "file body"},
        ],
    }


def test_translate_coalesces_consecutive_tool_results() -> None:
    """Two parallel tool calls → single user message with two tool_result blocks."""
    messages: list[Message] = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "a", "type": "function", "function": {"name": "t", "arguments": "{}"}},
                {"id": "b", "type": "function", "function": {"name": "t", "arguments": "{}"}},
            ],
        },
        {"role": "tool", "tool_call_id": "a", "content": "res A"},
        {"role": "tool", "tool_call_id": "b", "content": "res B"},
        {"role": "user", "content": "continue"},
    ]
    _, out = AnthropicCompatProvider._translate_messages(messages)
    # assistant block(s) → user(tool_results) → user(continue)
    assert out[1]["role"] == "user"
    assert len(out[1]["content"]) == 2
    assert out[1]["content"][0]["tool_use_id"] == "a"
    assert out[1]["content"][1]["tool_use_id"] == "b"
    assert out[2] == {"role": "user", "content": "continue"}


def test_translate_assistant_tool_call_with_invalid_json_preserves_raw() -> None:
    messages: list[Message] = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "x",
                    "type": "function",
                    "function": {"name": "t", "arguments": "{broken"},
                }
            ],
        },
    ]
    _, out = AnthropicCompatProvider._translate_messages(messages)
    tool_use = out[0]["content"][0]
    assert tool_use["input"] == {"_raw": "{broken"}


# ---------------------------------------------------------------------------
# Tool schema conversion
# ---------------------------------------------------------------------------

def test_to_anthropic_tool_maps_parameters_to_input_schema() -> None:
    schema = ToolSchema(
        name="read_file",
        description="Read a file",
        parameters={"type": "object", "properties": {"path": {"type": "string"}}},
        permission_level=PermissionLevel.READ_ONLY,
    )
    out = AnthropicCompatProvider._to_anthropic_tool(schema)
    assert out == {
        "name": "read_file",
        "description": "Read a file",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
        },
    }


# ---------------------------------------------------------------------------
# Stop-reason mapping
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "reason, has_tools, expected",
    [
        ("tool_use", True, "tool_calls"),
        ("end_turn", False, "stop"),
        ("max_tokens", False, "length"),
        ("stop_sequence", False, "stop"),
        (None, True, "tool_calls"),   # defensive: tools present but no reason
        (None, False, "stop"),
        ("weird", False, "weird"),   # passthrough for unknown
    ],
)
def test_map_stop_reason(reason, has_tools, expected) -> None:
    assert AnthropicCompatProvider._map_stop_reason(
        reason, has_tools=has_tools,
    ) == expected


# ---------------------------------------------------------------------------
# Stream parsing — fake the anthropic SDK stream context manager
# ---------------------------------------------------------------------------

class _FakeEvent:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _FakeStream:
    """Minimal async-context-manager matching ``messages.stream(...)``."""
    def __init__(self, events: list):
        self._events = events

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def __aiter__(self):
        async def gen():
            for e in self._events:
                yield e
        return gen()


class _FakeMessages:
    def __init__(self, events: list):
        self._events = events
        self.last_kwargs: dict | None = None

    def stream(self, **kwargs):
        self.last_kwargs = kwargs
        return _FakeStream(self._events)


class _FakeClient:
    def __init__(self, events: list):
        self.messages = _FakeMessages(events)


def _make_provider_with_events(events: list) -> AnthropicCompatProvider:
    p = AnthropicCompatProvider(api_key="test", model="claude-sonnet-4-5")
    p._client = _FakeClient(events)  # type: ignore[assignment]
    return p


@pytest.mark.asyncio
async def test_stream_emits_tokens_and_done_for_text_reply() -> None:
    events = [
        _FakeEvent(
            type="content_block_delta",
            index=0,
            delta=_FakeEvent(type="text_delta", text="Hello"),
        ),
        _FakeEvent(
            type="content_block_delta",
            index=0,
            delta=_FakeEvent(type="text_delta", text=" world"),
        ),
        _FakeEvent(
            type="message_delta",
            delta=_FakeEvent(stop_reason="end_turn"),
        ),
    ]
    p = _make_provider_with_events(events)
    chunks = [
        c async for c in p.stream_message([{"role": "user", "content": "hi"}])
    ]
    assert [c.type for c in chunks] == ["token", "token", "done"]
    assert chunks[0].data == "Hello"
    assert chunks[1].data == " world"
    assert chunks[2].data == "stop"


@pytest.mark.asyncio
async def test_stream_emits_tool_call_events_in_order() -> None:
    events = [
        _FakeEvent(
            type="content_block_start",
            index=0,
            content_block=_FakeEvent(type="tool_use", id="tu_1", name="read_file"),
        ),
        _FakeEvent(
            type="content_block_delta",
            index=0,
            delta=_FakeEvent(type="input_json_delta", partial_json='{"path":'),
        ),
        _FakeEvent(
            type="content_block_delta",
            index=0,
            delta=_FakeEvent(type="input_json_delta", partial_json='"foo.txt"}'),
        ),
        _FakeEvent(
            type="message_delta",
            delta=_FakeEvent(stop_reason="tool_use"),
        ),
    ]
    p = _make_provider_with_events(events)
    chunks = [
        c async for c in p.stream_message([{"role": "user", "content": "read foo"}])
    ]
    types = [c.type for c in chunks]
    assert types == [
        "tool_call_start",
        "tool_call_delta",
        "tool_call_delta",
        "tool_call_complete",
        "done",
    ]
    start = chunks[0].data
    assert start == {"id": "tu_1", "name": "read_file"}
    complete = chunks[3].data
    assert complete.id == "tu_1"
    assert complete.name == "read_file"
    assert complete.arguments == {"path": "foo.txt"}
    assert chunks[-1].data == "tool_calls"


@pytest.mark.asyncio
async def test_stream_emits_complete_on_content_block_stop() -> None:
    """Unit 4: emit tool_call_complete as soon as content_block_stop arrives,
    not at end of message — better UX with multiple tool calls per turn."""
    events = [
        _FakeEvent(
            type="content_block_start",
            index=0,
            content_block=_FakeEvent(type="tool_use", id="tu_1", name="read_file"),
        ),
        _FakeEvent(
            type="content_block_delta",
            index=0,
            delta=_FakeEvent(type="input_json_delta", partial_json='{"path":"a"}'),
        ),
        _FakeEvent(type="content_block_stop", index=0),
        _FakeEvent(
            type="content_block_start",
            index=1,
            content_block=_FakeEvent(type="tool_use", id="tu_2", name="read_file"),
        ),
        _FakeEvent(
            type="content_block_delta",
            index=1,
            delta=_FakeEvent(type="input_json_delta", partial_json='{"path":"b"}'),
        ),
        _FakeEvent(type="content_block_stop", index=1),
        _FakeEvent(type="message_delta", delta=_FakeEvent(stop_reason="tool_use")),
    ]
    p = _make_provider_with_events(events)
    chunks = [
        c async for c in p.stream_message([{"role": "user", "content": "go"}])
    ]
    types = [c.type for c in chunks]
    # Expect the first complete to land BEFORE the second start.
    first_complete_idx = types.index("tool_call_complete")
    second_start_idx = [i for i, t in enumerate(types) if t == "tool_call_start"][1]
    assert first_complete_idx < second_start_idx, (
        f"complete should come before next start, got {types}"
    )
    # Two completes total, no double-emit.
    assert types.count("tool_call_complete") == 2


@pytest.mark.asyncio
async def test_stream_handles_out_of_order_indices_via_id_keying() -> None:
    """Unit 4: accumulator is keyed by tool_use.id. If a provider reuses or
    swaps event.index between blocks (observed on MiniMax /anthropic), we
    still merge deltas into the right tool by following the id mapping."""
    events = [
        _FakeEvent(
            type="content_block_start",
            index=0,
            content_block=_FakeEvent(type="tool_use", id="tu_A", name="t"),
        ),
        _FakeEvent(
            type="content_block_start",
            index=1,
            content_block=_FakeEvent(type="tool_use", id="tu_B", name="t"),
        ),
        # Interleaved deltas — index=1 arrives before index=0's first chunk.
        _FakeEvent(
            type="content_block_delta", index=1,
            delta=_FakeEvent(type="input_json_delta", partial_json='{"x":'),
        ),
        _FakeEvent(
            type="content_block_delta", index=0,
            delta=_FakeEvent(type="input_json_delta", partial_json='{"y":'),
        ),
        _FakeEvent(
            type="content_block_delta", index=1,
            delta=_FakeEvent(type="input_json_delta", partial_json='2}'),
        ),
        _FakeEvent(
            type="content_block_delta", index=0,
            delta=_FakeEvent(type="input_json_delta", partial_json='1}'),
        ),
        _FakeEvent(type="message_delta", delta=_FakeEvent(stop_reason="tool_use")),
    ]
    p = _make_provider_with_events(events)
    chunks = [
        c async for c in p.stream_message([{"role": "user", "content": "go"}])
    ]
    completes = [c for c in chunks if c.type == "tool_call_complete"]
    assert len(completes) == 2
    by_id = {c.data.id: c.data for c in completes}
    assert by_id["tu_A"].arguments == {"y": 1}
    assert by_id["tu_B"].arguments == {"x": 2}


@pytest.mark.asyncio
async def test_stream_skips_orphan_delta_with_no_prior_start() -> None:
    """Unit 4: defensive — input_json_delta with an unknown index is dropped,
    not crashed on. Some misbehaving endpoints have been observed to send
    deltas without the preceding content_block_start."""
    events = [
        _FakeEvent(
            type="content_block_delta", index=99,
            delta=_FakeEvent(type="input_json_delta", partial_json='{"orphan":1}'),
        ),
        _FakeEvent(type="message_delta", delta=_FakeEvent(stop_reason="end_turn")),
    ]
    p = _make_provider_with_events(events)
    chunks = [
        c async for c in p.stream_message([{"role": "user", "content": "go"}])
    ]
    # No tool_call_* events at all — the orphan was silently dropped.
    types = [c.type for c in chunks]
    assert "tool_call_start" not in types
    assert "tool_call_delta" not in types
    assert "tool_call_complete" not in types
    assert types[-1] == "done"


@pytest.mark.asyncio
async def test_stream_passes_system_and_tools_to_sdk() -> None:
    events = [
        _FakeEvent(type="message_delta", delta=_FakeEvent(stop_reason="end_turn")),
    ]
    p = _make_provider_with_events(events)
    schema = ToolSchema(
        name="t",
        description="d",
        parameters={"type": "object"},
    )
    _ = [
        c async for c in p.stream_message(
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "u"},
            ],
            tools=[schema],
        )
    ]
    kwargs = p._client.messages.last_kwargs  # type: ignore[attr-defined]
    assert kwargs["system"] == "sys"
    assert kwargs["messages"] == [{"role": "user", "content": "u"}]
    assert kwargs["tools"] == [
        {"name": "t", "description": "d", "input_schema": {"type": "object"}},
    ]

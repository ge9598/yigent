"""Regression tests for Fix 1 — compression must not split tool_use pairs.

Layers 3, 4, and 5 all slice the conversation by index. Without guarding, the
slice boundary can fall between an assistant(tool_use) message and its
following tool(tool_result), orphaning the tool_result. Anthropic API rejects
that with HTTP 400.
"""

from __future__ import annotations

import pytest

from src.context.engine import CompressionEngine, _safe_split_index


class _StubAux:
    """Minimal aux provider stub — returns a canned summary text."""
    async def stream_message(self, messages, temperature=0.0, tools=None):
        class _Chunk:
            def __init__(self, type_, data):
                self.type = type_
                self.data = data
        yield _Chunk("token", "STUB_SUMMARY")
        yield _Chunk("done", None)


def _build_conv_with_tool_chains(n_turns: int) -> list[dict]:
    """Build a conversation where every other turn uses a tool.

    Shape per turn pair:
        user:      "q{i}"
        assistant: text + tool_calls=[tc_i]
        tool:      tool_call_id=tc_i, content="result_i"
    """
    msgs = []
    for i in range(n_turns):
        msgs.append({"role": "user", "content": f"q{i}"})
        tc_id = f"tc_{i}"
        msgs.append({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": tc_id,
                "function": {"name": "read_file", "arguments": '{"path":"/x"}'},
            }],
        })
        msgs.append({
            "role": "tool",
            "tool_call_id": tc_id,
            "name": "read_file",
            "content": f"result_{i}",
        })
    return msgs


def _assert_no_orphan_tool_results(msgs: list[dict]) -> None:
    """Every role=tool must be preceded (transitively, through consecutive
    tool siblings) by an assistant message whose tool_calls contain its id."""
    for i, m in enumerate(msgs):
        if m.get("role") != "tool":
            continue
        tool_call_id = m.get("tool_call_id")
        # Walk backward through any adjacent tool messages.
        j = i - 1
        while j >= 0 and msgs[j].get("role") == "tool":
            j -= 1
        assert j >= 0, f"tool_result at {i} has no preceding assistant"
        parent = msgs[j]
        assert parent.get("role") == "assistant", (
            f"tool_result at {i} preceded by {parent.get('role')!r}, "
            f"not assistant"
        )
        ids = {tc.get("id") for tc in (parent.get("tool_calls") or [])}
        assert tool_call_id in ids, (
            f"tool_result id={tool_call_id!r} not in preceding assistant's "
            f"tool_calls {ids}"
        )


# ---------------------------------------------------------------------------
# _safe_split_index unit tests
# ---------------------------------------------------------------------------

def test_safe_split_at_clean_boundary_returns_unchanged():
    msgs = [
        {"role": "user"}, {"role": "assistant"}, {"role": "user"},
    ]
    assert _safe_split_index(msgs, 1) == 1


def test_safe_split_walks_past_tool_message():
    msgs = [
        {"role": "user"},
        {"role": "assistant", "tool_calls": [{"id": "a"}]},
        {"role": "tool", "tool_call_id": "a"},
        {"role": "user"},
    ]
    # desired=2 would orphan the tool_result → walk forward to 3
    assert _safe_split_index(msgs, 2) == 3


def test_safe_split_walks_past_multiple_tool_messages():
    msgs = [
        {"role": "assistant", "tool_calls": [{"id": "a"}, {"id": "b"}]},
        {"role": "tool", "tool_call_id": "a"},
        {"role": "tool", "tool_call_id": "b"},
        {"role": "user"},
    ]
    assert _safe_split_index(msgs, 1) == 3
    assert _safe_split_index(msgs, 2) == 3


def test_safe_split_at_extremes():
    msgs = [{"role": "user"}, {"role": "assistant"}]
    assert _safe_split_index(msgs, 0) == 0
    assert _safe_split_index(msgs, 2) == 2


def test_safe_split_all_tool_returns_len():
    """If walking would exhaust, return len(msgs) so callers detect degenerate."""
    msgs = [
        {"role": "tool", "tool_call_id": "a"},
        {"role": "tool", "tool_call_id": "b"},
    ]
    assert _safe_split_index(msgs, 1) == len(msgs)


# ---------------------------------------------------------------------------
# Layer 3
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_layer3_does_not_orphan_tool_results():
    engine = CompressionEngine(auxiliary_provider=_StubAux())
    msgs = _build_conv_with_tool_chains(8)  # 24 msgs total
    out = await engine._layer3_summarize_early(msgs)
    _assert_no_orphan_tool_results(out)


# ---------------------------------------------------------------------------
# Layer 4
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_layer4_does_not_orphan_tool_results():
    engine = CompressionEngine(auxiliary_provider=_StubAux(), layer4_keep_turns=5)
    msgs = _build_conv_with_tool_chains(6)  # 18 msgs; layer4 keeps last 5
    out = await engine._layer4_rewrite(msgs)
    _assert_no_orphan_tool_results(out)


# ---------------------------------------------------------------------------
# Layer 5
# ---------------------------------------------------------------------------

def test_layer5_does_not_orphan_tool_results():
    engine = CompressionEngine(layer5_keep_turns=4)
    msgs = _build_conv_with_tool_chains(5)  # 15 body msgs; keep 4
    out = engine._layer5_hard_truncate(msgs)
    _assert_no_orphan_tool_results(out)


def test_layer5_degenerate_tail_all_tool_keeps_full_body(caplog):
    """If the last N body messages are all tool messages, keep full body."""
    import logging
    engine = CompressionEngine(layer5_keep_turns=2)
    msgs = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "a"}, {"id": "b"}, {"id": "c"}]},
        {"role": "tool", "tool_call_id": "a", "content": "r1"},
        {"role": "tool", "tool_call_id": "b", "content": "r2"},
        {"role": "tool", "tool_call_id": "c", "content": "r3"},
    ]
    with caplog.at_level(logging.WARNING):
        out = engine._layer5_hard_truncate(msgs)
    _assert_no_orphan_tool_results(out)
    # Keeps full body rather than emitting a 400-causing orphan split.
    assert len(out) == len(msgs)

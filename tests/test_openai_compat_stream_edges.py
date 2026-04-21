"""Regression tests for Fix 3 + Fix 4 — openai_compat streaming edges.

Fix 3: tool_call id and name must accumulate across chunks (Azure OpenAI and
some vLLM configs split them).

Fix 4: if the stream ends without a terminal finish_reason chunk (vLLM /
OpenRouter edge), accumulated tool_calls and reasoning must still be flushed.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.providers.openai_compat import OpenAICompatProvider


# ---------------------------------------------------------------------------
# Fake stream infrastructure
# ---------------------------------------------------------------------------

def _mk_tc_delta(index, id=None, name=None, arguments=None):
    fn = None
    if name is not None or arguments is not None:
        fn = SimpleNamespace(name=name, arguments=arguments)
    return SimpleNamespace(index=index, id=id, function=fn)


def _mk_chunk(*, content=None, tool_calls=None, finish_reason=None,
              reasoning_content=None):
    delta = SimpleNamespace(
        content=content,
        tool_calls=tool_calls,
        reasoning_content=reasoning_content,
    )
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice])


class _FakeStream:
    def __init__(self, chunks):
        self._chunks = chunks
    def __aiter__(self):
        return self._iter()
    async def _iter(self):
        for c in self._chunks:
            yield c


class _FakeCompletions:
    def __init__(self, chunks):
        self._chunks = chunks
    async def create(self, **kwargs):
        return _FakeStream(self._chunks)


class _FakeClient:
    def __init__(self, chunks):
        self.chat = SimpleNamespace(completions=_FakeCompletions(chunks))


def _build_provider(chunks) -> OpenAICompatProvider:
    p = OpenAICompatProvider(api_key="sk-x", base_url="https://x.test/v1",
                             model="test-model")
    p._client = _FakeClient(chunks)
    return p


async def _collect(p, messages=None):
    messages = messages or [{"role": "user", "content": "hi"}]
    return [c async for c in p.stream_message(messages=messages)]


# ---------------------------------------------------------------------------
# Fix 3 — tool_call id/name across chunks
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_tool_call_id_and_name_captured_across_chunks():
    """Chunk 1: id only. Chunk 2: name. Chunk 3+: arguments."""
    chunks = [
        _mk_chunk(tool_calls=[_mk_tc_delta(index=0, id="call_abc")]),
        _mk_chunk(tool_calls=[_mk_tc_delta(index=0, name="search_files")]),
        _mk_chunk(tool_calls=[_mk_tc_delta(index=0, arguments='{"q":"')]),
        _mk_chunk(tool_calls=[_mk_tc_delta(index=0, arguments='foo"}')]),
        _mk_chunk(finish_reason="tool_calls"),
    ]
    out = await _collect(_build_provider(chunks))
    starts = [c for c in out if c.type == "tool_call_start"]
    completes = [c for c in out if c.type == "tool_call_complete"]
    assert len(starts) == 1
    assert starts[0].data == {"id": "call_abc", "name": "search_files"}
    assert len(completes) == 1
    assert completes[0].data.id == "call_abc"
    assert completes[0].data.name == "search_files"
    assert completes[0].data.arguments == {"q": "foo"}


@pytest.mark.asyncio
async def test_tool_call_eager_id_and_name_still_works():
    """Classic case: chunk 1 has both id+name — behavior unchanged."""
    chunks = [
        _mk_chunk(tool_calls=[_mk_tc_delta(index=0, id="call_x", name="read_file",
                                          arguments='{"p":"/a"}')]),
        _mk_chunk(finish_reason="tool_calls"),
    ]
    out = await _collect(_build_provider(chunks))
    starts = [c for c in out if c.type == "tool_call_start"]
    completes = [c for c in out if c.type == "tool_call_complete"]
    assert len(starts) == 1
    assert starts[0].data == {"id": "call_x", "name": "read_file"}
    assert len(completes) == 1
    assert completes[0].data.name == "read_file"


@pytest.mark.asyncio
async def test_tool_call_start_not_emitted_until_both_id_and_name_present():
    """If only id arrives on chunk 1, don't emit a start with name=''."""
    chunks = [
        _mk_chunk(tool_calls=[_mk_tc_delta(index=0, id="call_abc")]),
        # No second chunk with name → never started; final flush still
        # includes arguments and emits tool_call_complete.
        _mk_chunk(tool_calls=[_mk_tc_delta(index=0, arguments='{}')]),
        _mk_chunk(finish_reason="tool_calls"),
    ]
    out = await _collect(_build_provider(chunks))
    starts = [c for c in out if c.type == "tool_call_start"]
    # Start was never emitted because name never arrived — but the complete
    # fallback still fires so the caller sees the tool call (with empty name,
    # which is at least honest).
    assert len(starts) == 0
    completes = [c for c in out if c.type == "tool_call_complete"]
    assert len(completes) == 1


# ---------------------------------------------------------------------------
# Fix 4 — stream ends without finish_reason
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_stream_without_finish_reason_still_flushes_tool_calls():
    """vLLM / OpenRouter may close the stream without terminal finish_reason."""
    chunks = [
        _mk_chunk(tool_calls=[_mk_tc_delta(index=0, id="call_x", name="read_file",
                                          arguments='{"p":"/a"}')]),
        # no finish_reason chunk
    ]
    out = await _collect(_build_provider(chunks))
    completes = [c for c in out if c.type == "tool_call_complete"]
    done = [c for c in out if c.type == "done"]
    assert len(completes) == 1
    assert completes[0].data.name == "read_file"
    assert len(done) == 1
    assert done[0].data == "incomplete"


@pytest.mark.asyncio
async def test_stream_without_finish_reason_still_flushes_reasoning():
    """Reasoning buffered via reasoning_content must survive abrupt close."""
    chunks = [
        _mk_chunk(reasoning_content="thinking..."),
        _mk_chunk(content="partial"),
        # no finish_reason
    ]
    out = await _collect(_build_provider(chunks))
    reasoning = [c for c in out if c.type == "reasoning"]
    assert len(reasoning) == 1
    assert reasoning[0].data["text"] == "thinking..."
